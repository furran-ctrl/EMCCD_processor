import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.quick_plot import plot_ndarray
from src.io.tiff_import import TiffLoader

Xray_test_data = EMCCDimage(TiffLoader(r'C:\Users\86177\Desktop\test_file\test_signal_600'
                                       ,'AndorEMCCD-36_xps189.140000_scan1_labtime22-54-33p894132'))

plot_ndarray(Xray_test_data.raw_data, 500, 1400)
#Xray_test_data.filter_xray_spots(sigma_threshold= 5.0)
