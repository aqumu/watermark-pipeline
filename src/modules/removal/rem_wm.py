import sys
import os
import cv2
import numpy as np

from ..base import BaseRemover
from src.utils.logger import logger

class RemWMRemover(BaseRemover):
    def __init__(self, rem_wm_path: str = "external/rem-wm", use_gpu: bool = False, resize_limit: int = 800):
        abs_path = os.path.abspath(rem_wm_path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
        
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config, HDStrategy, LDMSampler
        except ImportError as e:
            raise ImportError(f"Could not import lama_cleaner from {rem_wm_path}. Error: {e}")
            
        self.device = 'cuda' if use_gpu else 'cpu'
        logger.info(f"Initializing RemWMRemover (Lama Cleaner) on {self.device}")
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if use_gpu else '-1'
        self.model_manager = ModelManager(name="lama", device=self.device)
        self.Config = Config
        self.HDStrategy = HDStrategy
        self.LDMSampler = LDMSampler
        self.resize_limit = resize_limit
        
    def remove(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        config = self.Config(
            ldm_steps=15,
            ldm_sampler=self.LDMSampler.ddim,
            hd_strategy=self.HDStrategy.RESIZE,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=self.resize_limit,
            hd_strategy_resize_limit=self.resize_limit,
        )
        
        result = self.model_manager(image_rgb, mask, config)
        return result
