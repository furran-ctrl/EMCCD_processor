import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.result_filter import ResultsFilter
from pathlib import Path

# Initialize filter
filter_processor = ResultsFilter(r"C:\Users\86177\Desktop\0809\analysis_parallel_time")

print('initiated processor!')
# Process all XPS files
summary = filter_processor.process_all_xps_files()

# Or process individual file
#filter_processor.plot_distributions(Path("results/analyze_001/xps_188.94000/xps_188.94000.parquet"))