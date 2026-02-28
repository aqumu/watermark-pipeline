# Terms of Reference (ToR)

## Open-Source Image Processing Pipeline for Watermark Detection, Removal, and Optional Upscaling

------------------------------------------------------------------------

## 1. Project Overview

This project aims to design and implement a scalable, cost-effective,
open-source pipeline capable of processing large volumes of images to:

1.  Detect watermark location and scale using a known template image
2.  Remove the watermark reliably with minimal visual artifacts
3.  Optionally upscale images with acceptable quality improvements

The system must be suitable for batch processing workloads of at least
4,000 images per day, with architecture scalable to significantly higher
throughput if needed.

The solution must prioritize:

-   Open-source components
-   Cost-efficient computation (CPU-first, GPU optional)
-   Scalability and automation
-   Maintainability and reproducibility

------------------------------------------------------------------------

## 2. Objectives

### Primary Objectives

The pipeline must:

-   Detect watermark position and scale using a provided watermark
    template
-   Remove watermark using inpainting or equivalent reconstruction
    techniques
-   Support optional image upscaling using open-source super-resolution
    models
-   Support batch processing of large image datasets
-   Operate autonomously with minimal manual intervention

### Secondary Objectives

The pipeline should:

-   Support GPU acceleration when available
-   Support parallel processing across CPU cores and/or GPU devices
-   Allow flexible deployment (local, server, or cloud)
-   Be modular and extensible

------------------------------------------------------------------------

## 3. Functional Requirements

### 3.1 Watermark Detection

The system must detect watermark location even if:

-   Slightly offset from center
-   Scaled differently
-   Applied to images of varying resolution and aspect ratio

Preferred detection methods:

-   Template matching (OpenCV)
-   Feature matching (ORB, SIFT, or equivalent)
-   Optional neural network-based detection (if needed)

Output must include:

-   Bounding box coordinates
-   Watermark scale factor
-   Binary mask suitable for inpainting

------------------------------------------------------------------------

### 3.2 Watermark Removal

The system must remove the watermark using:

-   Mask-based inpainting
-   Template-based masking
-   Reconstruction methods that preserve surrounding image quality

Preferred open-source tools:


-   rem-wm
-   OpenCV inpainting (fallback option)

Output must:

-   Preserve original image resolution (unless upscaled)
-   Avoid visible artifacts where possible

------------------------------------------------------------------------

### 3.3 Image Upscaling (Optional)

The system must optionally upscale images when configured to do so.

Requirements:

-   Support scaling factors (e.g., 2×, 4×)
-   Maintain reasonable processing speed
-   Allow enable/disable via configuration

Preferred open-source tools:

-   Real-ESRGAN
-   OpenCV interpolation (fallback)

------------------------------------------------------------------------

### 3.4 Batch Processing

The system must support:

-   Processing entire directories
-   Recursive folder processing
-   Processing large image volumes without manual intervention

Supported formats:

-   JPEG
-   PNG
-   WEBP

------------------------------------------------------------------------

## 4. Non-Functional Requirements

### 4.1 Scalability

The system must support:

-   Multi-threaded CPU processing
-   Multi-process parallelism
-   GPU acceleration (CUDA)
-   Distributed processing (optional future extension)

Target scaling capability:

-   Minimum: 4,000 images/day
-   Target: 100,000+ images/day per deployment instance

------------------------------------------------------------------------

### 4.2 Computational Efficiency

The system must prioritize:

-   CPU efficiency for cost reduction
-   GPU acceleration as optional performance enhancement
-   Batch processing optimization
-   Efficient memory usage

Target processing times:

-   CPU mode: ≤ 500 ms per image (watermark removal only)
-   GPU mode: ≤ 100 ms per image (watermark removal only)

------------------------------------------------------------------------

### 4.3 Cost Efficiency

The system must be deployable on:

-   Low-cost CPU servers
-   Optional GPU servers for higher throughput

Target deployment cost range:

-   CPU deployment: low monthly operating cost
-   GPU deployment: optional for higher performance scaling

------------------------------------------------------------------------

### 4.4 Modularity

The system must be modular, consisting of independent components:

-   Input module
-   Detection module
-   Mask generation module
-   Watermark removal module
-   Upscaling module
-   Output module

Modules must be replaceable independently.

------------------------------------------------------------------------

## 5. Recommended Open-Source Components

Preferred primary components:

### Watermark removal

-   rem-wm (optional alternative)
-   OpenCV inpainting (fallback)

### Detection

-   OpenCV template matching
-   ORB/SIFT feature matching

### Upscaling

-   Real-ESRGAN (primary)
-   OpenCV resize (fallback)

### Processing framework

-   Python 3.10+
-   NumPy
-   OpenCV
-   PyTorch (for AI models)

------------------------------------------------------------------------

## 6. System Architecture

### Logical pipeline flow

Input Images → Watermark Detection → Mask Generation → Watermark Removal
→ Optional Upscaling → Output Images

### Scalable deployment architecture (optional production setup)

Image Storage → Processing Queue → Worker Nodes (CPU/GPU) → Output
Storage

------------------------------------------------------------------------

## 7. Hardware Requirements

### Minimum (CPU deployment)

-   4 CPU cores
-   8 GB RAM
-   No GPU required

### Recommended (scalable deployment)

-   8+ CPU cores
-   16+ GB RAM
-   Optional NVIDIA GPU with CUDA support

------------------------------------------------------------------------

## 8. Deliverables

Required deliverables include:

-   Fully functional pipeline implementation
-   Configuration file for system settings
-   Installation instructions
-   Usage documentation
-   Test dataset validation results

Optional deliverables:

-   Docker container
-   Automated batch processing script

------------------------------------------------------------------------

## 9. Acceptance Criteria

The system will be considered complete when it:

-   Successfully detects watermark location in ≥ 95% of test images
-   Removes watermark with acceptable visual quality
-   Successfully processes batch datasets automatically
-   Supports optional upscaling
-   Operates reliably in CPU mode
-   Supports GPU acceleration if available

------------------------------------------------------------------------

## 10. Constraints

The solution must:

-   Use open-source software only
-   Not depend on proprietary APIs
-   Run locally or on self-hosted infrastructure

------------------------------------------------------------------------

## 11. Future Extensions (Optional)

Potential future enhancements:

-   Distributed processing cluster
-   Automatic watermark detection without template
-   Quality assessment automation
-   REST API interface
-   Web-based control interface

------------------------------------------------------------------------

## 12. Summary

This project will deliver a scalable, cost-efficient, open-source
pipeline capable of detecting, removing, and optionally upscaling
watermarked images using modern inpainting and super-resolution
techniques.

The system will support both CPU-only and GPU-accelerated deployments
and will be suitable for large-scale automated image processing
workflows.
