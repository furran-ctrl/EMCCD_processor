import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from analyzer_code.filter.center_coor_filter import CenterCoordinateFilter
from analyzer_code.filter.radial_bin_analyzer import RadialBinAnalyzer

# Step 1: Filter center coordinates
#center_filter = CenterCoordinateFilter(r"C:\Users\86177\Desktop\0809\analysis_parallel_time")
#filter_results = center_filter.process_all_files()

# Step 2: Analyze specific XPS group
# Initialize with custom export directory
radial_analyzer = RadialBinAnalyzer(export_base_dir=r"C:\Users\86177\Desktop\0809\analysis_parallel_time")
analysis_results = radial_analyzer.process_all_xps_groups(
    analysis_directory=r"C:\Users\86177\Desktop\0809\analysis_parallel_time"
)