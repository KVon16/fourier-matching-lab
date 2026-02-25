"""Utilities for Fourier-transform intuition practice.

Core goals:
- Keep transforms consistent across CLI and web app.
- Render magnitude images in an exam-like grayscale style.
- Support real texture images and synthetic pattern generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage import data


@dataclass
class StyleConfig:
    clip_percentile: float = 99.5
    gamma: float = 3.0
    gain: float = 1.81
    grain_std: float = 0.05
    grain_seed: int = 442
    suppress_dc_radius: int = 0


REAL_IMAGE_BUILDERS = {
    "brick": data.brick,
    "grass": data.grass,
    "gravel": data.gravel,
    "coins": data.coins,
    "checkerboard": data.checkerboard,
    "camera": data.camera,
    "moon": data.moon,
    "page": data.page,
}


def load_images_from_folder(folder: str | Path, size: int = 512) -> Dict[str, np.ndarray]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    out: Dict[str, np.ndarray] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[p.stem] = center_crop_square(img, size=size)

    if not out:
        raise ValueError(f"No readable image files in {folder}")
    return out


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def center_crop_square(image: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = image[y0 : y0 + side, x0 : x0 + side]
    if side != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped


def load_real_images(size: int = 512, names: List[str] | None = None) -> Dict[str, np.ndarray]:
    selected = names or list(REAL_IMAGE_BUILDERS.keys())
    out: Dict[str, np.ndarray] = {}
    for name in selected:
        if name not in REAL_IMAGE_BUILDERS:
            raise ValueError(f"Unknown real image '{name}'. Choices: {sorted(REAL_IMAGE_BUILDERS)}")
        arr = REAL_IMAGE_BUILDERS[name]()
        arr = _to_gray_uint8(arr)
        arr = center_crop_square(arr, size=size)
        out[name] = arr
    return out


def synthetic_patterns(size: int = 512) -> Dict[str, np.ndarray]:
    y, x = np.indices((size, size))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    freq = 0.045
    stripes = (np.sin(2 * np.pi * freq * x) > 0).astype(np.float32)

    checker = (((x // 24 + y // 24) % 2) * 255).astype(np.uint8)

    radial = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    rings = (np.sin(2 * np.pi * 0.03 * radial) > 0).astype(np.float32)

    diagonal = (np.sin(2 * np.pi * (0.03 * x + 0.03 * y)) > 0).astype(np.float32)

    out = {
        "stripes": (255 * stripes).astype(np.uint8),
        "checker": checker,
        "rings": (255 * rings).astype(np.uint8),
        "diagonal": (255 * diagonal).astype(np.uint8),
    }
    return out


def fft_magnitude_exam_style(image: np.ndarray, style: StyleConfig | None = None) -> np.ndarray:
    style = style or StyleConfig()
    gray = _to_gray_uint8(image)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Optional tiny DC suppression avoids a saturated white center dot.
    if style.suppress_dc_radius > 0:
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        rr = style.suppress_dc_radius
        fshift[cy - rr : cy + rr + 1, cx - rr : cx + rr + 1] = 0

    mag = np.log1p(np.abs(fshift))

    # Robust clipping keeps bright spikes while preserving low-amplitude grain.
    hi = np.percentile(mag, style.clip_percentile)
    mag = np.clip(mag / max(hi, 1e-6), 0.0, 1.0)
    mag = np.power(mag, style.gamma)
    mag = np.clip(mag * style.gain, 0.0, 1.0)

    if style.grain_std > 0:
        rng = np.random.default_rng(style.grain_seed)
        mag = np.clip(mag + rng.normal(0, style.grain_std, size=mag.shape), 0.0, 1.0)

    return (mag * 255).astype(np.uint8)


def pairwise_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_fft_features(mag: np.ndarray) -> Dict[str, float]:
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    row_energy = float(np.mean(mag[cy - 1 : cy + 2, :]))
    col_energy = float(np.mean(mag[:, cx - 1 : cx + 2]))

    yy, xx = np.indices(mag.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radial_mid = mag[(rr > h * 0.15) & (rr < h * 0.35)]
    radial_outer = mag[(rr > h * 0.35) & (rr < h * 0.48)]

    return {
        "center_row_energy": row_energy,
        "center_col_energy": col_energy,
        "mid_radius_energy": float(np.mean(radial_mid)) if radial_mid.size else 0.0,
        "outer_radius_energy": float(np.mean(radial_outer)) if radial_outer.size else 0.0,
    }


def annotate_intuition(features: Dict[str, float]) -> str:
    row = features["center_row_energy"]
    col = features["center_col_energy"]
    mid = features["mid_radius_energy"]
    outer = features["outer_radius_energy"]

    hints: List[str] = []
    if col > row * 1.1:
        hints.append("Strong vertical line in spectrum -> image has mostly horizontal repetition.")
    elif row > col * 1.1:
        hints.append("Strong horizontal line in spectrum -> image has mostly vertical repetition.")
    else:
        hints.append("Balanced cross energy -> both horizontal and vertical structures are present.")

    if outer > mid * 1.05:
        hints.append("More outer energy -> sharper edges/finer detail.")
    else:
        hints.append("More mid-frequency energy -> broader repetitive blocks.")

    return " ".join(hints)


def build_dataset(
    use_real: bool = True,
    size: int = 512,
    style: StyleConfig | None = None,
    folder: str | Path | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if folder is not None:
        images = load_images_from_folder(folder=folder, size=size)
    else:
        images = load_real_images(size=size) if use_real else synthetic_patterns(size=size)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, img in images.items():
        mag = fft_magnitude_exam_style(img, style=style)
        out[name] = (img, mag)
    return out
