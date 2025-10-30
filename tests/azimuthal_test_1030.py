import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.mask_class import RingMask, precompute_ring_mask
from src.core.mask_class import RadialMasks, precompute_radial_masks, precompute_azimuthal_average_masks
from src.io.tiff_import import TiffLoader
from src.core.tiff_objects import EMCCDimage
from src.core.bkg_objects import CrudeBackground
from src.io.quick_plot import plot_ndarray

Bkgtestdir = r"C:\Users\86177\Desktop\test_file\bkg_100"
OldBkg = CrudeBackground(Bkgtestdir)
CrudeBkg = OldBkg.get_processed_background()

testdata = EMCCDimage(TiffLoader(r'E:\20250808\8_water_IR72deg_longscan5\fist_AndorEMCCD',
                                    'AndorEMCCD-1353_xps188.945000_scan4_labtime23-08-45p717613'))

testdata.remove_background(CrudeBkg)
RingMask_ = precompute_ring_mask()
testdata.iterative_ring_centroid(RingMask_, [720,350])

plot_ndarray(testdata.get_processed_data(),300,600)
RadialMasks_ = precompute_radial_masks(radius=512, num_bins=512)
azimuthal_dict = precompute_azimuthal_average_masks(RadialMasks_, (1024, 1024))

a, I_avg = testdata.azimuthal_average_bincount(RadialMasks_,azimuthal_dict)
print(type(I_avg))
print(I_avg)