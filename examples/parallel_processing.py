import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.full_directory_processor import DirectoryProcessor

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=r"C:\Users\ab177\Desktop\diffraction_results\1005day",
    data_directory=r"E:\20251005\day2_61deg_longscan7\fist_AndorEMCCD",
    xps_grouping_param=[250, 0.002],  # [threshold, tolerance]
    xray_removal_param=[15, 0.7],  # [beam_threshold, expansion_threshold_ratio]
    center_fitting_param=[80, 180, 540, 495],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[512, 512],  # [radius, num_bins]
    background_directory="default",
    data_mask_directory="default"
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=8, analyze_no="analysis_parallel_longscan7")

# Load existing configuration
#processor.load_config("analysis_001")

#.\venv\Scripts\Activate
# git config --global user.email "xsr23@mails.tsinghua.edu.cn"
# git config --global user.name "Theory"