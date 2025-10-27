#x=720 y=350
import sys
import os
import numpy as np
from typing import Dict, Tuple

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader

import time
from contextlib import contextmanager

@contextmanager
def timer(description: str = "代码块"):
    """
    计时上下文管理器
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{description} 执行时间: {end - start:.4f} 秒")

processed = EMCCDimage(TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace',r'data_saved2.tiff'))
processed.copy_as_processed()

with timer('gaussian'):
    processed.find_diffraction_center([100,740,370,100,5],100,300)

with timer('centroid'):
    print(processed.ring_centroid([740,370]))

with timer('i_centroid'):
    print(processed.iterative_ring_centroid([740,370]))

def test_func_for_annotation(data: np.ndarray, 
                     center: Tuple[float, float],
                     radius: float = 300,
                     num_bins: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate azimuthal average of 2D array around specified center.
    
    Args:
        data: Input 2D numpy array (1024, 1024)
        center: Center coordinates as (x, y) tuple
        radius: Maximum radius for averaging in pixels. Defaults to 300.
        num_bins: Number of radial bins. Defaults to 300.
    
    Returns:
        tuple: Contains two arrays:
            - radial_positions: Bin center coordinates (num_bins,)
            - average_intensities: Azimuthally averaged intensities (num_bins,)
    
    Raises:
        ValueError: If data is not 2D array or center is outside bounds
        TypeError: If input types are incorrect
    
    Example:
        >>> data = np.random.rand(1024, 1024)
        >>> center = (512, 512)
        >>> radii, intensities = azimuthal_average(data, center)
        >>> print(f"Peak at radius: {radii[np.argmax(intensities)]}")
    """
    return
