import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from pathlib import Path
from analyzer_code.filter.center_coor_filter import CenterCoordinateFilter
from analyzer_code.filter.radial_bin_analyzer import RadialBinAnalyzer

analyze_directory = r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking\analysis_parallel_time"

# Step 1: Filter center coordinates
center_filter = CenterCoordinateFilter(analyze_directory)
filter_results = center_filter.process_all_files()

# Step 2: Analyze in detail with baseline
# Initialize with custom export directory
radial_analyzer = RadialBinAnalyzer(export_base_dir=analyze_directory)
analysis_results = radial_analyzer.process_all_xps_groups(
    analysis_directory=analyze_directory
)

# Analyze single 
# filepath = r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking\analysis_parallel_time\xps_221.02480\filtered_xps_221.02480.parquet"
# file_path = Path(filepath)
# df = pd.read_parquet(file_path)
# Analyze this XPS group
# summary_df = radial_analyzer.analyze_single_xps(df, 221.02480, file_path.parent)