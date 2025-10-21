import tifffile as tiff
import numpy as np
from typing import Optional, Tuple, List

class TiffLoader:
    """Handles loading of tiff image formats"""
    
    @staticmethod
    def load_tiff(filepath: str) -> np.ndarray:
        return tiff.imread(filepath)
    
    @staticmethod
    def load_image_stack(filepaths: List[str]) -> List[np.ndarray]:
        return [ImageLoader.load_tiff(fp) for fp in filepaths]