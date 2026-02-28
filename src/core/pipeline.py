import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..modules.base import BaseDetector, BaseRemover, BaseUpscaler
from ..utils.io import load_image, save_image, get_image_files

logger = logging.getLogger("WatermarkPipeline.Core")

class WatermarkPipeline:
    """
    Main orchestrator for the watermark processing pipeline.
    Combines independent modules for detection, removal, and optional upscaling.
    """
    def __init__(
        self,
        detector: BaseDetector,
        remover: BaseRemover,
        upscaler: BaseUpscaler | None = None
    ):
        self.detector = detector
        self.remover = remover
        self.upscaler = upscaler

    def process_image(self, input_path: Path, output_path: Path) -> bool:
        """Processes a single image through the pipeline."""
        try:
            image = load_image(input_path)
            if image is None:
                logger.error(f"Failed to load image: {input_path}")
                return False
                
            mask, detected = self.detector.detect(image)
            
            if detected:
                logger.debug(f"Watermark detected in {input_path.name}. Removing...")
                result_image = self.remover.remove(image, mask)
            else:
                logger.debug(f"No watermark detected in {input_path.name}. Copying original.")
                result_image = image
                
            if self.upscaler is not None:
                logger.debug(f"Upscaling image: {input_path.name}")
                result_image = self.upscaler.upscale(result_image)
                
            success = save_image(output_path, result_image)
            if success:
                logger.info(f"Successfully processed: {input_path.name}")
            else:
                logger.error(f"Failed to save image to {output_path}")
                
            return success
        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {str(e)}", exc_info=True)
            return False

    def process_directory(self, input_dir: str | Path, output_dir: str | Path, num_workers: int = 4) -> int:
        """Processes an entire directory in parallel."""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        
        files = get_image_files(in_path)
        if not files:
            logger.warning(f"No usable images found in '{in_path}'.")
            return 0
            
        logger.info(f"Found {len(files)} image(s) to process. Utilizing {num_workers} worker(s).")
        success_count = 0
        
        # Fallback to synchronous if num_workers is 1 or less
        if num_workers <= 1:
            for file_path in files:
                rel_path = file_path.relative_to(in_path)
                out_file = out_path / rel_path
                if self.process_image(file_path, out_file):
                    success_count += 1
            return success_count
            
        # Parallel processing for batch execution using ThreadPoolExecutor
        # This avoids pickling issues with Torch models on Windows (spawn)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for file_path in files:
                rel_path = file_path.relative_to(in_path)
                out_file = out_path / rel_path
                futures[executor.submit(self.process_image, file_path, out_file)] = file_path
                
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Unhandled exception processing {file_path.name}: {e}")
                    
        return success_count
