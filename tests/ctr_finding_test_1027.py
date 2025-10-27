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
