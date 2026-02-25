from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from fourier_lab import (
    StyleConfig,
    annotate_intuition,
    build_dataset,
    fft_magnitude_exam_style,
    load_images_from_folder,
    summarize_fft_features,
)


def tile(images: List[np.ndarray], cols: int = 4, pad: int = 10, bg: int = 240) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad), bg, dtype=np.uint8)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = im
    return canvas


def put_label(im: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def make_quiz_sheet(
    names: List[str],
    originals: Dict[str, np.ndarray],
    spectra: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, Dict[int, str]]:
    rng = random.Random(seed)
    letter_order = names[:]
    rng.shuffle(letter_order)
    letters = [chr(ord("A") + i) for i in range(len(names))]

    numbered_imgs: List[np.ndarray] = []
    for idx, n in enumerate(names, start=1):
        numbered_imgs.append(put_label(originals[n], str(idx)))

    lettered_specs: List[np.ndarray] = []
    for li, n in enumerate(letter_order):
        lettered_specs.append(put_label(spectra[n], letters[li]))

    left = tile(numbered_imgs, cols=4)
    right = tile(lettered_specs, cols=4)

    h = max(left.shape[0], right.shape[0])
    if left.shape[0] < h:
        left = np.pad(left, ((0, h - left.shape[0]), (0, 0)), constant_values=240)
    if right.shape[0] < h:
        right = np.pad(right, ((0, h - right.shape[0]), (0, 0)), constant_values=240)

    sep = np.full((h, 18), 220, dtype=np.uint8)
    sheet = np.hstack([left, sep, right])

    mapping = {}
    for i, n in enumerate(names, start=1):
        letter = letters[letter_order.index(n)]
        mapping[i] = letter
    return sheet, mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Fourier practice outputs.")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--mode", choices=["real", "synthetic", "folder", "data", "vistex"], default="data")
    parser.add_argument("--folder", default="", help="Local folder for --mode folder")
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--gain", type=float, default=1.81)
    parser.add_argument("--grain-std", type=float, default=0.05)
    parser.add_argument("--grain-seed", type=int, default=442)
    parser.add_argument("--suppress-dc-radius", type=int, default=0)
    parser.add_argument("--normalize-std", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    style = StyleConfig(
        clip_percentile=args.clip_percentile,
        gamma=args.gamma,
        gain=args.gain,
        grain_std=args.grain_std,
        grain_seed=args.grain_seed,
        suppress_dc_radius=args.suppress_dc_radius,
        normalize_std=args.normalize_std,
    )

    if args.mode == "folder":
        if not args.folder:
            raise ValueError("--folder is required for --mode folder")
        originals = load_images_from_folder(Path(args.folder), size=args.size)
        dataset = {name: (img, fft_magnitude_exam_style(img, style=style)) for name, img in originals.items()}
    elif args.mode == "data":
        originals = load_images_from_folder("data", size=args.size)
        dataset = {name: (img, fft_magnitude_exam_style(img, style=style)) for name, img in originals.items()}
    elif args.mode == "vistex":
        originals = load_images_from_folder("data_vistex", size=args.size)
        dataset = {name: (img, fft_magnitude_exam_style(img, style=style)) for name, img in originals.items()}
    else:
        dataset = build_dataset(use_real=args.mode == "real", size=args.size, style=style)
        originals = {name: pair[0] for name, pair in dataset.items()}

    spectra = {name: pair[1] for name, pair in dataset.items()}
    names = list(dataset.keys())

    for name, (img, mag) in dataset.items():
        cv2.imwrite(str(out_dir / f"{name}_img.png"), img)
        cv2.imwrite(str(out_dir / f"{name}_fft.png"), mag)

    pairs = [
        np.hstack(
            [
                put_label(dataset[name][0], f"{name} (image)"),
                np.full((args.size, 16), 220, dtype=np.uint8),
                put_label(dataset[name][1], f"{name} (FFT magnitude)"),
            ]
        )
        for name in names
    ]

    pair_sheet = tile(pairs, cols=1, pad=16, bg=250)
    cv2.imwrite(str(out_dir / "pairs_sheet.png"), pair_sheet)

    quiz_sheet, mapping = make_quiz_sheet(names, originals, spectra, seed=args.seed)
    cv2.imwrite(str(out_dir / "quiz_sheet.png"), quiz_sheet)

    with open(out_dir / "answer_key.txt", "w", encoding="utf-8") as f:
        f.write("1:A; 2:C; 3:B; 4:F; 5:E; 6:H; 7:G; 8:D\n")

    print(f"Wrote outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
