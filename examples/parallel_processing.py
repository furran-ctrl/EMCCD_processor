import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path
from scripts.full_directory_processor import DirectoryProcessor

Expdata_folder = Path(r"E:\20251012\night3_longscan_uv63p4_IR37\fist_AndorEMCCD")
Result_folder = Path(r"C:\Users\ab177\Desktop\diffraction_results\test")

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=Result_folder,
    data_directory=Expdata_folder,
    xps_grouping_param=[600, 0.002],  # [threshold, tolerance]
    xray_removal_param=[15, 0.7],  # [beam_threshold, expansion_threshold_ratio]
    center_fitting_param=[40, 120, 705, 727],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[512, 512],  # [radius, num_bins]
    background_directory="default",
    data_mask_directory="default"
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=1, analyze_no="analysis_center_test")

# Load existing configuration
#processor.load_config("analysis_001")

#.\venv\Scripts\Activate
# git config --global user.email "xsr23@mails.tsinghua.edu.cn"
# git config --global user.name "Theory"