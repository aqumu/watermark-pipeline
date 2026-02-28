# Watermark Detection and Removal Pipeline

An open-source image processing pipeline for detecting and removing watermarks from high volumes of images, with optional upscaling support.

## Architecture
- **`src/core/pipeline.py`**: Orchestrates loading, detection, removal, optional upscaling, and saving.
- **`src/modules/detection/watermark_anything.py`**: AI watermark localization using Facebook Research's `watermark-anything` model.
- **`src/modules/removal/rem_wm.py`**: Inpainting/removal through `rem-wm` (`lama_cleaner`) using the detector-generated mask.
- **`config.yaml`**: Data-driven configuration to swap modules and tune thresholds/runtime settings.

## Detection and Removal Flow
1. Input image is passed to **Watermark Anything** for mask prediction.
2. The predicted mask is upsampled back to original image size and binarized.
3. The binary mask is normalized to `uint8` `[0, 255]` before being passed to **rem-wm**.
4. `rem-wm` inpaints watermark regions and returns the cleaned image.

## Requirements
- Python 3.10+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup notes
- Ensure `external/watermark-anything` exists (the pipeline expects this path by default).
- On first run, the detector downloads `wam_mit.pth` automatically to `external/watermark-anything/checkpoints/` if it is missing.
- Ensure `external/rem-wm` is available and install its own dependencies.

## Usage
Run the pipeline with default configuration (`config.yaml`):
```bash
python main.py
```

Provide a custom configuration file:
```bash
python main.py --config custom_config.yaml
```

Override specific parameters via CLI:
```bash
python main.py --input ./data/input --output ./data/output --workers 8
```
