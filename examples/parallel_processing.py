import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.full_directory_processor import DirectoryProcessor

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking",
    data_directory=r"D:\20250926\3_longscan5_44deg\fist_AndorEMCCD",
    xps_grouping_param=[700, 0.002],  # [threshold, tolerance]
    xray_removal_param=[18, 0.7],  # [beam_threshold, expansion_threshold_ratio]
    center_fitting_param=[40, 200, 681, 440],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[512, 512],  # [radius, num_bins]
    background_directory="default",
    data_mask_directory="default"
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=8, analyze_no="analysis_parallel_time")

# Load existing configuration
#processor.load_config("analysis_001")