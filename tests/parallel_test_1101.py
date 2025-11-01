import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from scripts.full_directory_processor import DirectoryProcessor

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=r"C:\Users\86177\Desktop\results",
    data_directory=r"E:\20250808\8_water_IR72deg_longscan6\fist_AndorEMCCD",
    background_directory="default",
    xps_grouping_param=[350, 0.002],  # [threshold, tolerance]
    xray_removal_param=[32, 29.6, 300],  # [chunk_size, sigma_threshold, beam_threshold]
    center_fitting_param=[40, 200, 720, 350],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[512, 512]  # [radius, num_bins]
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=8, analyze_no="analysis_parallel_time")

# Load existing configuration
#processor.load_config("analysis_001")