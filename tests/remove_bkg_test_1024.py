import sys
import os
import numpy as np
from typing import Dict, Tuple

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.io.quick_plot import plot_ndarray
#from src.core.bkg_objects import CrudeBackground

r'''
bkgdata = TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace',r'Temp_savespace.tiff')
if bkgdata is None:
    print('Wrong Datatype!!')
Bkgtestdir = r"C:\Users\86177\Desktop\test_file\bkg_100"
BkgTest = CrudeBackground(Bkgtestdir)
CrudeBkg = BkgTest.get_processed_background()
'''
bkgdata = TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace',r'calc_bkg.tiff')

data_filepath = r"C:\Users\86177\Desktop\test_file\test_signal_600"
data_filename = r"AndorEMCCD-5_xps188.975000_scan1_labtime22-54-14p065387.tiff"
tiffdata = EMCCDimage(TiffLoader(data_filepath,data_filename))
print(r'loaded')
tiffdata.remove_background(bkgdata)
print(r"calced")
#plot_ndarray(CrudeBkg,470,520)
#plot_ndarray(bkgdata,470,520)
plot_ndarray(tiffdata.get_processed_data(),20,300)

from src.io.result_save import save_as_tiff
result_dir = r"C:\Users\86177\Desktop\Diffraction_code\Temp_savespace\data_saved5"
#save_as_tiff(tiffdata.get_processed_data(),result_dir)
#X小Y大远离暗电流
'''
def detailed_corner_analysis(array: np.ndarray, corner_size: int = 50) -> Dict:
    """
    详细的角落区域分析，返回统计信息
    """
    # 定义角落区域坐标
    corners = {
        'top_left': (0, corner_size, 0, corner_size),
        'top_right': (0, corner_size, -corner_size, None),
        'bottom_left': (-corner_size, None, 0, corner_size),
        'bottom_right': (-corner_size, None, -corner_size, None)
    }
    
    results = {}
    
    for corner_name, (row_start, row_end, col_start, col_end) in corners.items():
        # 提取角落区域
        corner_region = array[row_start:row_end, col_start:col_end]
        
        # 计算统计信息
        results[corner_name] = {
            'mean': np.mean(corner_region),
            'std': np.std(corner_region),
            'min': np.min(corner_region),
            'max': np.max(corner_region),
            'shape': corner_region.shape,
            'coordinates': f"rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]"
        }
    
    return results

# 使用示例
analysis = detailed_corner_analysis(tiffdata.get_processed_data(),30)
for corner, stats in analysis.items():
    print(f"\n{corner.upper()}:")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  区域形状: {stats['shape']}")


'''
from scipy import ndimage
def gaussian_filter(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    高斯滤波 - 最常用的低通滤波
    
    Parameters:
    -----------
    array : np.ndarray
        输入数组 (1024, 1024)
    sigma : float
        高斯核的标准差，控制平滑程度
    
    Returns:
    --------
    np.ndarray : 滤波后的数组
    """
    return ndimage.gaussian_filter(array, sigma=sigma, mode='reflect') 
def mean_filter(array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    均值滤波 - 最简单的低通滤波
    
    Parameters:
    -----------
    array : np.ndarray
        输入数组 (1024, 1024)
    kernel_size : int
        滤波核大小，必须是奇数
    
    Returns:
    --------
    np.ndarray : 滤波后的数组
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须是奇数")
    
    # 使用均匀滤波核
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # 应用滤波
    filtered = ndimage.convolve(array, kernel, mode='reflect')
    
    return filtered

import matplotlib.pyplot as plt
def plot_x_slice_average(array: np.ndarray, x_start: int = 0, x_end: int = 200):
    """
    计算x方向指定区域在y方向的平均值并绘图
    
    Parameters:
    -----------
    array : np.ndarray
        输入数组 (1024, 1024)
    x_start : int
        x方向起始位置
    x_end : int
        x方向结束位置
    """
    # 验证输入范围
    if x_start < 0 or x_end > array.shape[1] or x_start >= x_end:
        raise ValueError("无效的x范围")
    
    # 提取x=0~200区域
    x_slice = array[:, x_start:x_end]
    
    # 计算y方向的平均值 (对x方向求平均)
    y_profile = np.mean(x_slice, axis=1)
    y_std = np.std(x_slice, axis=1)
    # y坐标
    y_coords = np.arange(array.shape[0])
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_coords, y_std, 'b-', linewidth=1)
    plt.xlabel('Y Position (pixels)')
    plt.ylabel('Average Intensity')
    plt.title(f'Y-direction Profile (averaged over x={x_start}~{x_end})')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = np.mean(y_profile)
    std_val = np.std(y_profile)
    plt.axhline(y=mean_val, color='r', linestyle='--', 
                label=f'Mean: {mean_val:.2f} ± {std_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return y_profile, y_coords

#plot_x_slice_average(tiffdata.get_processed_data(),0,80)
#plot_ndarray(mean_filter(tiffdata.get_processed_data(),3),10,30)   