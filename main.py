import argparse
import sys
import yaml
from pathlib import Path

from src.utils.logger import logger
from src.modules.detection.template_matcher import TemplateMatcherDetector
from src.core.pipeline import WatermarkPipeline

def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_pipeline(config: dict) -> WatermarkPipeline:
    """Builds the WatermarkPipeline instance parsing the provided config dict."""
    
    # 1. Setup Detector module
    det_config = config.get("detection", {})
    if det_config.get("method") == "template_matching":
        detector = TemplateMatcherDetector(
            template_path=det_config.get("template_path", "data/template.png"),
            threshold=det_config.get("threshold", 0.8)
        )
    else:
        raise ValueError(f"Unknown detection method configured: {det_config.get('method')}")

    # 2. Setup Remover module
    rem_config = config.get("removal", {})
    if rem_config.get("method") == "rem-wm":
        from src.modules.removal.rem_wm import RemWMRemover
        remover = RemWMRemover(
            rem_wm_path=rem_config.get("path", "external/rem-wm"),
            use_gpu=rem_config.get("use_gpu", False),
            resize_limit=rem_config.get("resize_limit", 800)
        )
    else:
        raise ValueError(f"Unknown removal method configured: {rem_config.get('method')}")

    # 3. Setup Upscaler module
    up_config = config.get("upscaling", {})
    upscaler = None
    if up_config.get("enabled", False):
        if up_config.get("method") == "real-esrgan":
            from src.modules.upscaling.real_esrgan import RealESRGANUpscaler
            upscaler = RealESRGANUpscaler(
                realesrgan_path=up_config.get("path", "external/Real-ESRGAN"),
                model_name=up_config.get("model_name", "RealESRGAN_x4plus"),
                scale_factor=up_config.get("scale_factor", 4.0),
                use_gpu=up_config.get("use_gpu", False),
                tile=up_config.get("tile", 0),
                tile_pad=up_config.get("tile_pad", 10)
            )
        else:
            raise ValueError(f"Unknown upscaling method configured: {up_config.get('method')}")

    return WatermarkPipeline(detector, remover, upscaler)

def parse_args():
    parser = argparse.ArgumentParser(description="Watermark Removal Scalable Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML configuration file")
    
    # Allow overriding basic pipeline parameters directly via CLI
    parser.add_argument('--input', type=str, help="Override input directory")
    parser.add_argument('--output', type=str, help="Override output directory")
    parser.add_argument('--workers', type=int, help="Override number of workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)
        
    logger.info("Initializing Modular Watermark Pipeline...")
    
    try:
        # Build independent modules based on config
        pipeline = build_pipeline(config)
        
        # Extract pipeline execution variables
        pipe_config = config.get("pipeline", {})
        
        # Prioritize command-line arguments if provided, else rely on config
        input_dir = args.input if args.input else pipe_config.get("input_dir", "data/input")
        output_dir = args.output if args.output else pipe_config.get("output_dir", "data/output")
        num_workers = args.workers if args.workers is not None else pipe_config.get("num_workers", 4)
        
        logger.info(f"Target paths -> Input: '{input_dir}' | Output: '{output_dir}'")
        
        success_count = pipeline.process_directory(
            input_dir=input_dir, 
            output_dir=output_dir,
            num_workers=num_workers
        )
        
        logger.info(f"Pipeline flow completed. Successful assets processed: {success_count}.")
        
    except Exception as e:
        logger.error(f"Critical error executing pipeline: {e}", exc_info=False)
        sys.exit(1)

if __name__ == "__main__":
    main()
