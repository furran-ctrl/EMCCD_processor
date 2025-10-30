import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.quick_plot import plot_ndarray
from src.io.tiff_import import TiffLoader

Xray_test_data = EMCCDimage(TiffLoader(r'E:\20250808\8_water_IR72deg_longscan5\fist_AndorEMCCD'
                                       ,'AndorEMCCD-1353_xps188.945000_scan4_labtime23-08-45p717613'))

bkg_data = TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace'
                      , 'calc_bkg')

Xray_test_data.remove_background(bkg_data)

plot_ndarray(Xray_test_data.get_processed_data(), 70, 1200)
#Xray_test_data.filter_xray_spots(sigma_threshold= 5.0)
