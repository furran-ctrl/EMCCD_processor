import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 检查目录是否存在
#src_path = os.path.join(project_root, 'src')
#core_path = os.path.join(project_root, 'src', 'core')
#print(f"src目录存在: {os.path.exists(src_path)}")
#print(f"core目录存在: {os.path.exists(core_path)}")

from src.core.bkg_objects import CrudeBackground

Bkgtestdir = r"C:\Users\86177\Desktop\test_file\bkg_100"
BkgTest = CrudeBackground(Bkgtestdir)
CrudeBkg = BkgTest.get_processed_background()

import matplotlib.pyplot as plt
import numpy as np

def plot_background(Bkg: np.ndarray):

    plt.figure(figsize=(10, 8))
    # Create the plot
    im = plt.imshow(Bkg, cmap='viridis', aspect='equal') 
    # Add colorbar
    plt.colorbar(im, label='Intensity')
    
    # Add labels and title
    plt.title('Background Image (1024*1024)')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Usage
plot_background(CrudeBkg)

from src.io.result_save import save_as_tiff

result_dir = r"C:\Users\86177\Desktop\Diffraction_code\Temp_savespace\calc_bkg"
save_as_tiff(CrudeBkg,result_dir)
