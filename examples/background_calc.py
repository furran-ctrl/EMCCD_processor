import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.bkg_objects import CrudeBackground, CalcBackground
from src.io.quick_plot import plot_ndarray
from src.io.result_save import save_as_tiff

Bkgtestdir = r"C:\Users\86177\Desktop\test_file\bkg_100"
#directory for background .tiffs
'''
OldBkg = CrudeBackground(Bkgtestdir)
CrudeBkg = OldBkg.get_processed_background()
#Currently calculating mean to show the difference
'''

NewBkg = CalcBackground(Bkgtestdir)
FineBkg = NewBkg.process_background(chunk_size= 32, sigma_threshold= 22, beam_threshold= 1000)

plot_ndarray(FineBkg, 420, 650)
Bkgdir = r'C:\Users\86177\Desktop\results\background'
#directory under the related processing folder
save_as_tiff(FineBkg, Bkgdir)