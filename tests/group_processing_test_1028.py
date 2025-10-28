import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.XPSBatch_Process import BatchXPSProcessor

# Example usage
batch_processor = BatchXPSProcessor(
    data_directory="/path/to/tiff/files",
    bkg_directory="/path/to/background",
    bkg_filename="background.tiff",
    initial_center_guess=(720, 350)
)

# Organize files into XPS groups
batch_processor.organize_files_into_groups(threshold=350, tolerance=0.002)

# Process first 2 groups for testing
results = batch_processor.process_groups_sequential(process_first_n=2)

# Get processing summary
summary = batch_processor.get_processing_summary()
print("Batch processing summary:", summary)