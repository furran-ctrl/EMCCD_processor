import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analyzer_code.filter.radial_filter import RadialProfileFilter
from analyzer_code.filter.diffraction_normalizer import DiffractionNormalizer
from analyzer_code.plotting.newPDwaterfall import plot_waterfall_diffraction
from analyzer_code.plotting.PDFwaterfall import *
from analyzer_code.utils.Iavg_loader import load_intensity_profiles
from analyzer_code.utils.parquet_merge import merge_similar_xps_files

data_list = [r"C:\Users\ab177\Desktop\diffraction_results\1004\analysis_parallel_time",
             r"C:\Users\ab177\Desktop\diffraction_results\1005day\analysis_parallel_longscan3",
             r"C:\Users\ab177\Desktop\diffraction_results\1005day\analysis_parallel_longscan4",
             r"C:\Users\ab177\Desktop\diffraction_results\1005night\analysis_parallel_time"]
analyze_directory = r"C:\Users\ab177\Desktop\diffraction_results\1012long\analysis_parallel_time"

# merge_similar_xps_files(source_dirs=data_list,
#                         output_dir=analyze_directory,
#                         tolerance=0.0025,
#                         merge_method="mean")

# # Initialize the filter
# filter_processor = RadialProfileFilter(analyze_directory)

# # Run the filtering process
# results = filter_processor.run_filtering(filter_type="MAD")

# # Check results
# for xps_dir, result in results.items():
#     if result['success']:
#         print(f"{xps_dir}: Processed successfully")
#     else:
#         print(f"{xps_dir}: Failed - {result['message']}")

# # Initialize the normalizer
# normalizer = DiffractionNormalizer(analyze_directory)

# # Run the normalization process
# results = normalizer.run_normalization(statistic_type="normal")

# Generate summary report
#summary = normalizer.generate_summary_report(results)

# Check individual results
# for filename, result in results.items():
#     if result['success']:
#         print(f"{filename}: Processed successfully")
#     else:
#         print(f"{filename}: Failed - {result['message']}")

# Load all intensity profiles
intensity_data = load_intensity_profiles(analyze_directory, 
                                         std_check=True,
                                         calibration_factor=0.024,
                                         xps_0=198.08)

# Execute the function with the mock data
# plot_waterfall_diffraction(
#     data=intensity_data, 
#     scale=0.8, 
#     width_to_height_ratio=0.4,
#     filename=r"C:\Users\ab177\Desktop\waterfall_plot.png"
# )

s = np.array(intensity_data[0][2])
f = import_DCS_interpolated(20,3.7,s)
Ih2o = f[1]*f[1]*2+f[8]*f[8]

plot_waterfall_PDF(
    Iat=Ih2o,
    data=intensity_data, 
    scale=0.5, 
    width_to_height_ratio=0.4,
    filename=r"C:\Users\ab177\Desktop\PDF_plot.png"
)

# import matplotlib.pyplot as plt

# for k in range(1,20,1):
#     plt.plot(s, f[k+1]/f[8])
# plt.show()