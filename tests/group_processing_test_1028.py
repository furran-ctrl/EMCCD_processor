import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.XPSBatch_Process import BatchXPSProcessor

# Example usage
batch_processor = BatchXPSProcessor(
    data_directory=r"E:\20250808\8_water_IR72deg_longscan5\fist_AndorEMCCD",
    bkg_directory=r"C:\Users\86177\Desktop\Diffraction_code\Temp_savespace",
    bkg_filename="calc_bkg.tiff",
    azimuthal_radius=400,
    initial_center_guess=(720, 350)
)

# Organize files into XPS groups
batch_processor.organize_files_into_groups(threshold=20, tolerance=0.002)
print('organized')
# Process first 2 groups for testing
results = batch_processor.process_groups_sequential(process_first_n=1)

# Get processing summary
summary = batch_processor.get_processing_summary()
print("Batch processing summary:", summary)