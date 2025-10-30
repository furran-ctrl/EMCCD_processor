import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.quick_plot import plot_ndarray
from src.io.tiff_import import TiffLoader

Xray_test_data = EMCCDimage(TiffLoader(r'C:\Users\86177\Desktop\test_file\bkg_100'
                                       ,'AndorEMCCD-12_noxps0_labtime08-56-32p994206'))

Xray_test_data.copy_as_processed()
Xray_test_data.filter_xray_spots_inplace(sigma_threshold= 14,beam_threshold= 1000)

plot_ndarray(Xray_test_data.get_processed_data(), 400, 600)
#Xray_test_data.filter_xray_spots(sigma_threshold= 5.0)