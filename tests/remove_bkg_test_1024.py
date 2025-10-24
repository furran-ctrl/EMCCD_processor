import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.io.quick_plot import plot_ndarray

bkgdata = TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace',r'Temp_savespace.tiff')
if bkgdata is None:
    print('Wrong Datatype!!')

data_filepath = r"C:\Users\86177\Desktop\test_file\bkg_100"
data_filename = r"AndorEMCCD-4_noxps0_labtime08-56-27p879496.tiff"
tiffdata = EMCCDimage(TiffLoader(data_filepath,data_filename))

tiffdata.remove_background(bkgdata)

plot_ndarray(tiffdata.get_processed_data(),0,50)
