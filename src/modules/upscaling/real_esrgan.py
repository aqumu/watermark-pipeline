import sys
import os
import cv2
import numpy as np
from pathlib import Path

from ..base import BaseUpscaler
from src.utils.logger import logger

class RealESRGANUpscaler(BaseUpscaler):
    def __init__(self, realesrgan_path: str = "external/Real-ESRGAN", model_name: str = 'RealESRGAN_x4plus', scale_factor: float = 4.0, use_gpu: bool = False, tile: int = 0, tile_pad: int = 10):
        abs_path = os.path.abspath(realesrgan_path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
        except ImportError as e:
            raise ImportError(f"Could not import realesrgan from {realesrgan_path}. Error: {e}")
            
        self.scale_factor = scale_factor
        
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        else:
            raise ValueError(f"Model {model_name} is not fully configured in this wrapper.")

        model_path = os.path.join(abs_path, 'weights', model_name + '.pth')
        if not os.path.exists(model_path):
            os.makedirs(os.path.join(abs_path, 'weights'), exist_ok=True)
            logger.info(f"Downloading RealESRGAN model to {model_path}...")
            load_file_from_url(url=file_url, model_dir=os.path.join(abs_path, 'weights'), progress=True, file_name=None)
            
        device = 'cuda' if use_gpu else 'cpu'
        logger.info(f"Initializing RealESRGANUpscaler on {device} (tile={tile})")
        
        # RealESRGANer gpu_id parameter decides if it uses GPU
        gpu_id = None if use_gpu else -1
        
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=use_gpu, # fp16 only if GPU is used
            gpu_id=gpu_id
        )
        
    def upscale(self, image: np.ndarray) -> np.ndarray:
        try:
            output, _ = self.upsampler.enhance(image, outscale=self.scale_factor)
            return output
        except RuntimeError as e:
            logger.error(f"Error during upscaling: {e}")
            return image
