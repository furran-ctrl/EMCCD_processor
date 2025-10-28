import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProcessedResult:
    """存储单个文件处理结果的类"""
    filename: str
    xps_value: float
    center: Tuple[float, float]
    radial_profile: np.ndarray
    bin_centers: np.ndarray

@dataclass
class XPSGroupResult:
    """存储XPS组平均结果的类"""
    xps_value: float
    bin_centers: np.ndarray
    avg_radial_profile: np.ndarray
    std_radial_profile: np.ndarray
    file_count: int
    centers: List[Tuple[float, float]]