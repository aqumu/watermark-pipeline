import cv2
import numpy as np
from pathlib import Path

def load_image(filepath: str | Path) -> np.ndarray | None:
    """Load an image from disk."""
    return cv2.imread(str(filepath))

def save_image(filepath: str | Path, image: np.ndarray) -> bool:
    """Save an image to disk, creating parent directories if needed."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), image)

def get_image_files(directory: str | Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> list[Path]:
    """Retrieve a list of image files in a directory recursively."""
    in_path = Path(directory)
    if not in_path.exists() or not in_path.is_dir():
        return []
    
    files = []
    for file_path in in_path.rglob('*.*'):
        if file_path.suffix.lower() in extensions:
            files.append(file_path)
    return files
