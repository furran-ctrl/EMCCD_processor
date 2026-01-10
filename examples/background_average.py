import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from EMCCD_processor.processor_code.io.tiff_save import save_as_tiff
from processor_code.io.tiff_import import load_as_tiff

bkg3 = load_as_tiff(r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking","background3.tiff")
bkg1 = load_as_tiff(r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking","background1.tiff")
avgbkg = bkg3*0.25 + bkg1*0.75

Bkgdir = r'C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking\background'
save_as_tiff(avgbkg, Bkgdir)