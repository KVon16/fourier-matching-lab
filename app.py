from __future__ import annotations

import random
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st

from fourier_lab import (
    StyleConfig,
    annotate_intuition,
    build_dataset,
    fft_magnitude_exam_style,
    summarize_fft_features,
)

st.set_page_config(page_title="Fourier Matching Lab", layout="wide")
st.title("Fourier Matching Lab")
st.caption("Practice matching image patterns to their Fourier magnitude signatures.")

with st.sidebar:
    st.header("Dataset")
    source = st.radio(
        "Image source",
        [
            "Data folder (8 uploaded)",
            "MIT VisTex textures (16)",
            "Real textures",
            "Synthetic patterns",
            "Upload image",
        ],
        index=0,
    )
    size = st.slider("Working size", min_value=64, max_value=768, value=256, step=64)
    images_per_row = st.slider("Images per row", min_value=2, max_value=6, value=4, step=1)
    st.header("FFT Style")
    clip_percentile = st.slider("Clip percentile", min_value=99.0, max_value=100.0, value=99.5, step=0.01)
    gamma = st.slider("Gamma", min_value=1.0, max_value=3.0, value=3.0, step=0.05)
    gain = st.slider("Brightness gain", min_value=0.5, max_value=2.0, value=1.81, step=0.01)
    grain_std = st.slider("Grain", min_value=0.0, max_value=0.3, value=0.05, step=0.002)
    suppress_dc_radius = st.slider("Center suppression radius", min_value=0, max_value=8, value=0, step=1)
    grain_seed = st.number_input("Grain seed", min_value=0, max_value=10_000, value=442, step=1)

style = StyleConfig(
    clip_percentile=clip_percentile,
    gamma=gamma,
    gain=gain,
    grain_std=grain_std,
    grain_seed=int(grain_seed),
    suppress_dc_radius=suppress_dc_radius,
)


def draw_labeled(im: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_dataset(dataset: Dict[str, tuple[np.ndarray, np.ndarray]], per_row: int) -> None:
    cols = st.columns(per_row)
    for i, (name, (img, mag)) in enumerate(dataset.items()):
        with cols[i % per_row]:
            st.markdown(f"**{name}**")
            st.image(img, use_container_width=True, clamp=True)
            st.image(mag, use_container_width=True, clamp=True)
            feats = summarize_fft_features(mag)
            st.caption(annotate_intuition(feats))


if source == "Upload image":
    up = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if up is not None:
        file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error("Could not decode image.")
            st.stop()
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mag = fft_magnitude_exam_style(img, style=style)

        c1, c2 = st.columns(2)
        c1.image(img, caption="Input image", use_container_width=True)
        c2.image(mag, caption="FFT magnitude (exam style)", use_container_width=True)

        st.subheader("Intuition hint")
        feats = summarize_fft_features(mag)
        st.write(annotate_intuition(feats))
        st.json(feats)
        st.code(
            f"StyleConfig(clip_percentile={style.clip_percentile:.2f}, gamma={style.gamma:.2f}, "
            f"gain={style.gain:.2f}, grain_std={style.grain_std:.3f}, grain_seed={style.grain_seed}, "
            f"suppress_dc_radius={style.suppress_dc_radius})",
            language="python",
        )
    else:
        st.info("Upload an image to begin.")

else:
    if source == "Data folder (8 uploaded)":
        dataset = build_dataset(size=size, style=style, folder="data")
    elif source == "MIT VisTex textures (16)":
        dataset = build_dataset(size=size, style=style, folder="data_vistex")
    else:
        dataset = build_dataset(use_real=source == "Real textures", size=size, style=style)
    names = list(dataset.keys())

    tab1, tab2 = st.tabs(["Explore", "Matching quiz"])

    with tab1:
        show_dataset(dataset, per_row=images_per_row)

    with tab2:
        if "quiz_seed" not in st.session_state:
            st.session_state.quiz_seed = 442
        if st.button("Reshuffle"):
            st.session_state.quiz_seed = random.randint(0, 10_000)

        rng = random.Random(st.session_state.quiz_seed)
        letter_order: List[str] = names[:]
        rng.shuffle(letter_order)
        letters = [chr(ord("A") + i) for i in range(len(names))]
        letter_map = {letters[i]: letter_order[i] for i in range(len(names))}

        st.markdown("### Numbered images")
        ncols = st.columns(images_per_row)
        for i, name in enumerate(names, start=1):
            with ncols[(i - 1) % images_per_row]:
                st.image(draw_labeled(dataset[name][0], str(i)), use_container_width=True)

        st.markdown("### Lettered Fourier magnitudes")
        lcols = st.columns(images_per_row)
        for i, letter in enumerate(letters):
            with lcols[i % images_per_row]:
                st.image(draw_labeled(dataset[letter_map[letter]][1], letter), use_container_width=True)

        with st.expander("Show answer key"):
            st.code("1:A; 2:C; 3:B; 4:F; 5:E; 6:H; 7:G; 8:D")

st.sidebar.markdown("### Current params")
st.sidebar.code(
    f"StyleConfig(clip_percentile={style.clip_percentile:.2f}, gamma={style.gamma:.2f}, "
    f"gain={style.gain:.2f}, grain_std={style.grain_std:.3f}, grain_seed={style.grain_seed}, "
    f"suppress_dc_radius={style.suppress_dc_radius})",
    language="python",
)
