# Watermark Detection and Removal Pipeline

An open-source image processing pipeline for detecting and removing watermarks from high volumes of images, with optional upscaling support. This project has been engineered to be highly modular, scalable, and maintainable, satisfying the long-term vision described in the Terms of Reference while providing a functional MVP using OpenCV.

## Architecture & Scalability
The project is set up using a modular architecture that separates detection, removal, and upscaling into independent, interchangeable components.

- **`src/core/pipeline.py`**: The central orchestrator that manages loading, processing, and saving images. It natively uses `concurrent.futures.ProcessPoolExecutor` to ensure high throughput and utilize multi-core CPUs for batch processing.
- **`src/modules/`**: Contains base interfaces (`BaseDetector`, `BaseRemover`, `BaseUpscaler`) and their respective implementations. 
- **`config.yaml`**: The pipeline is fully data-driven. Using a configuration file makes deploying, tweaking parameters, and swapping out models (e.g., from OpenCV to AI models) seamless.

## Minimum Viable Product (MVP) Implementations
The current "basics" focus on establishing the end-to-end framework, powered by OpenCV fallbacks to minimize dependencies:
- **Detection**: Template Matching (`TemplateMatcherDetector`), optimized with scale pyramids.
- **Removal**: Telea inpainting (`OpenCVInpaintRemover`) as a CPU-based restoration method.
- **Upscaling**: OpenCV bicubic interpolation (`OpenCVUpscaler`).

*The abstraction leaves explicit room to easily slot in models like `lama-cleaner` (LaMa) or `Real-ESRGAN` in the future without rewiring the application logic.*

## Requirements
- Python 3.10+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline with the default configuration (`config.yaml`):
```bash
python main.py
```

Provide a custom configuration file:
```bash
python main.py --config custom_config.yaml
```

Override specific config parameters via CLI:
```bash
python main.py --input ./data/input --output ./data/output --workers 8
```
