import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.bkg_objects import CrudeBackground, CalcBackground
from src.io.quick_plot import plot_ndarray

Bkgtestdir = r"C:\Users\86177\Desktop\test_file\bkg_100"
OldBkg = CrudeBackground(Bkgtestdir)
CrudeBkg = OldBkg.get_processed_background()

NewBkg = CalcBackground(Bkgtestdir)
FineBkg = NewBkg.process_background(chunk_size= 32, sigma_threshold= 22, beam_threshold= 1000)

plot_ndarray(FineBkg, 420, 650)