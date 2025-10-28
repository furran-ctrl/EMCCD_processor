import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProcessedResult:
    """save result for single processed tiff"""
    #filename: str
    #xps_value: float
    center: Tuple[float, float]
    radial_profile: np.ndarray
    #bin_centers: np.ndarray

@dataclass
class XPSGroupResult:
    """save result for a xps group"""
    xps_value: float
    #bin_centers: np.ndarray
    all_radial_profile: List[np.ndarray]
    avg_radial_profile: np.ndarray
    std_radial_profile: np.ndarray
    file_count: int
    centers: List[Tuple[float, float]]
    total_counts: List[float]