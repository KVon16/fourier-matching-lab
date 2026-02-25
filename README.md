# Fourier Matching Practice Lab

This project helps you build intuition for matching image patterns to their Fourier transform magnitudes (like CV exam questions).

## What you get

- `make_fourier_set.py`: Python script to generate Fourier outputs and matching quiz sheets.
- `app.py`: Streamlit web app for interactive exploration and matching practice.
- `fourier_lab.py`: Shared logic for FFT magnitude styling and intuition hints.

The FFT magnitude rendering is tuned to an exam-like grayscale style:
- `fftshift` centered spectrum
- `log(1 + |F|)` compression
- robust percentile clipping
- gamma shaping for bright center + visible spikes
- optional grain texture for a worksheet-like look

Default style is intentionally darker (less washed out). You can tune:
- `clip_percentile`
- `gamma`
- `gain` (overall brightness)
- `grain_std`
- `suppress_dc_radius`

Optional pre-filtering (off by default):
- `normalize_std` to equalize contrast before FFT

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Generate Fourier sheets from Python

### Your 8 uploaded images in `data/` (default)
```bash
python make_fourier_set.py --mode data --out outputs
```

### Additional real textures from MIT VisTex (`data_vistex/`)
```bash
python make_fourier_set.py --mode vistex --out outputs_vistex
```

### Built-in real textures
```bash
python make_fourier_set.py --mode real --out outputs_real
```

Example with custom style:
```bash
python make_fourier_set.py --mode real --out outputs_custom_style \
  --clip-percentile 99.97 --gamma 1.35 --gain 0.62 --grain-std 0.018 --suppress-dc-radius 2
```

Example with pre-filtering (without changing default slider settings):
```bash
python make_fourier_set.py --mode vistex --out outputs_vistex_prefilter \
  --normalize-std
```

### Synthetic fallback
```bash
python make_fourier_set.py --mode synthetic --out outputs_synth
```

### Your own folder of real images
```bash
python make_fourier_set.py --mode folder --folder /absolute/path/to/images --out outputs_custom
```

Outputs include:
- `pairs_sheet.png`
- `quiz_sheet.png`
- `answer_key.txt`
- individual `*_img.png` and `*_fft.png`

## 2) Run the web app

```bash
streamlit run app.py
```

Use the app to:
- use your `data/` image set directly via `Lecture folder (8 uploaded)`,
- switch to `MIT VisTex textures (16)` for more real texture practice,
- inspect real textures and their FFT signatures,
- upload your own image,
- tune FFT rendering with live sliders and copy the exact `StyleConfig(...)`,
- reshuffle a numbered-vs-lettered matching quiz,
- reveal answer key only when ready.

## Midterm intuition checklist

- Strong **vertical line** in spectrum -> horizontal repetition in image.
- Strong **horizontal line** in spectrum -> vertical repetition in image.
- **Grid textures** -> cross-like peaks (both axes).
- **Diagonal structures** -> diagonal spectral streaks.
- **Sharp edges / fine detail** -> more high-frequency (outer) energy.
- **Large smooth blobs** -> low-frequency concentration near center.
