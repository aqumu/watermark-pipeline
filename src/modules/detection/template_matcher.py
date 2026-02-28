import cv2
import numpy as np
import logging

from ..base import BaseDetector

logger = logging.getLogger("WatermarkPipeline.TemplateMatcher")

class TemplateMatcherDetector(BaseDetector):
    def __init__(self, template_path: str, threshold: float = 0.8):
        self.template_path = template_path
        self.threshold = threshold
        
        self.template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Failed to read template image: {template_path}")
            
        self.template_h, self.template_w = self.template.shape

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray_image.shape
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        detected = False
        best_match = None
        
        # Test scales from 0.1 to 1.0 down to smaller sizes
        for scale in np.linspace(0.1, 1.0, 20)[::-1]:
            curr_w = int(self.template_w * scale)
            curr_h = int(self.template_h * scale)
            
            if curr_h > img_h or curr_w > img_w or curr_h < 10 or curr_w < 10:
                continue
                
            resized_template = cv2.resize(self.template, (curr_w, curr_h), interpolation=cv2.INTER_AREA)
            
            # Template matching
            res = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if best_match is None or max_val > best_match['val']:
                best_match = {
                    'val': max_val,
                    'scale': scale,
                    'loc': max_loc,
                    'w': curr_w,
                    'h': curr_h,
                    'template': resized_template
                }
                
        if best_match is not None and best_match['val'] >= self.threshold:
            logger.debug(f"Watermark found at scale {best_match['scale']:.2f}, conf {best_match['val']:.2f}")
            pt = best_match['loc']
            w = best_match['w']
            h = best_match['h']
            
            # Threshold the template to only mask the actual text/logo pixels
            _, template_mask = cv2.threshold(best_match['template'], 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # If the template background is white, invert the mask
            if best_match['template'].mean() > 127:
                template_mask = cv2.bitwise_not(template_mask)
            
            # Dilate mask for better inpainting edges
            kernel = np.ones((5, 5), np.uint8)
            template_mask = cv2.dilate(template_mask, kernel, iterations=1)
            
            # Apply to output mask
            mask_roi = mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
            cv2.bitwise_or(mask_roi, template_mask, dst=mask_roi)
            detected = True
        else:
            if best_match:
                logger.debug(f"Best match was {best_match['val']:.2f} (below threshold {self.threshold})")
                 
        return mask, detected
