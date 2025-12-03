import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analyzer_code.filter.radial_filter import RadialProfileFilter
from analyzer_code.filter.diffraction_normalizer import DiffractionNormalizer

analyze_directory = r"C:\Users\86177\Desktop\streaking\analysis_parallel_time"
# # Initialize the filter
# filter_processor = RadialProfileFilter(analyze_directory)

# # Run the filtering process
# results = filter_processor.run_filtering()

# # Check results
# for xps_dir, result in results.items():
#     if result['success']:
#         print(f"{xps_dir}: Processed successfully")
#     else:
#         print(f"{xps_dir}: Failed - {result['message']}")

# Initialize the normalizer
# normalizer = DiffractionNormalizer(analyze_directory)

# # Run the normalization process
# results = normalizer.run_normalization()

# Generate summary report
#summary = normalizer.generate_summary_report(results)

# Check individual results
# for filename, result in results.items():
#     if result['success']:
#         print(f"{filename}: Processed successfully")
#     else:
#         print(f"{filename}: Failed - {result['message']}")

from analyzer_code.plotting.newPDwaterfall import plot_waterfall_diffraction_final
from analyzer_code.utils.Iavg_loader import load_intensity_profiles

# Load all intensity profiles
intensity_data = load_intensity_profiles(analyze_directory)
# Execute the function with the mock data
plot_waterfall_diffraction_final(
    data=intensity_data, 
    xps_0=221.0075, 
    pixel_to_q=0.024, 
    scale=30, 
    width_to_height_ratio=0.4,
    filename=r"C:\Users\86177\Desktop\streaking\waterfall_plot_output.png"
)