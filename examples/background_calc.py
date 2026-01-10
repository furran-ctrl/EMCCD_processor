import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from processor_code.core.bkg_objects import CalcBackground
from processor_code.io.quick_plot import plot_ndarray
from processor_code.io.tiff_save import save_as_tiff

Bkgdatadir = r"E:\20251013\day1_uvoff_IR37\fist_AndorEMCCD"
#directory for background .tiffs

Bkgobj = CalcBackground(Bkgdatadir)
FineBkg = Bkgobj.process_background_Xray(sigma_threshold= 15, expansion_threshold_ratio= 0.7, center_pos=[0,0,0])

plot_ndarray(FineBkg, 420, 650)
Bkgsavedir = r"C:\Users\ab177\Desktop\diffraction_results\test\background"
#directory under the related processing folder
save_as_tiff(FineBkg, Bkgsavedir)
