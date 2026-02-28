from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Detect watermark in the image.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            mask (np.ndarray): Binary mask representing the watermark area.
            detected (bool): Whether a watermark was found.
        """
        pass

class BaseRemover(ABC):
    @abstractmethod
    def remove(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove watermark based on mask.
        
        Args:
            image (np.ndarray): The input image with watermark.
            mask (np.ndarray): The binary mask indicating watermark pixels.
            
        Returns:
            image (np.ndarray): The image with the watermark removed.
        """
        pass

class BaseUpscaler(ABC):
    @abstractmethod
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale the given image.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            image (np.ndarray): The upscaled image.
        """
        pass
