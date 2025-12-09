import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analyzer_code.filter.radial_filter import RadialProfileFilter
from analyzer_code.filter.diffraction_normalizer import DiffractionNormalizer
from analyzer_code.plotting.newPDwaterfall import plot_waterfall_diffraction
from analyzer_code.utils.Iavg_loader import load_intensity_profiles
from analyzer_code.utils.parquet_merge import merge_similar_xps_files

data_list = [r"C:\Users\ab177\Desktop\diffraction_results\1004\analysis_parallel_time",
             r"C:\Users\ab177\Desktop\diffraction_results\1005day\analysis_parallel_longscan3",
             r"C:\Users\ab177\Desktop\diffraction_results\1005day\analysis_parallel_longscan4",
             r"C:\Users\ab177\Desktop\diffraction_results\1005night\analysis_parallel_time",]

data_list1 = [r"C:\Users\ab177\Desktop\diffraction_results\1004\analysis_parallel_time",
             r"C:\Users\ab177\Desktop\diffraction_results\1005day\analysis_parallel_longscan4"]

merge_data = []
for i in range(4):
# Load all intensity profiles
    intensity_data = load_intensity_profiles(data_list[i], 
                                            std_check=True,
                                            calibration_factor=0.024,
                                            xps_0=172.435)
    #data structure:List[Tuple[float, np.ndarray, np.ndarray]]
    #choose the two groups with lowest timestamp as background
    bkg1 = intensity_data[0]
    bkg2 = intensity_data[1]
    print(bkg1[0],bkg2[0])
    #calculate the average background
    avg_bkg = (bkg1[1] + bkg2[1]) / 2
    #subtract background from each group other than the two background groups
    corrected_data = []
    for idx, data in enumerate(intensity_data):
        if idx < 2:
            corrected_data.append(data)
            continue
        #corrected_intensity = data[1] - avg_bkg
        #corrected_data.append((data[0], corrected_intensity, data[2]))
    merge_data.extend(corrected_data)
#if the data[0] is closer than 0.01, we consider them as the same group and merge them

# final_data = []
# tolerance = 0.01
# for i in range(1):
#     xps_value = merge_data[i][0]
#     merged_intensity = merge_data[i][1]
#     merged_std = merge_data[i][2]
#     count = 1
#     for j in range(1, i):
#         if abs(merge_data[j][0] - xps_value) < tolerance:
#             merged_intensity += merge_data[j][1]
#             #merged_std += merge_data[j][2]
#             count += 1
#             # Mark merged entry to avoid re-processing
#             del merge_data[j] 
#     merged_intensity /= count
#     #merged_std /= count
#     final_data.append((xps_value, merged_intensity, merged_std))
# print(f"Total groups after merging: {len(final_data)}")

# Execute the function with the mock data
plot_waterfall_diffraction(
    data=merge_data, 
    scale=10, 
    width_to_height_ratio=0.8,
    filename=r"C:\Users\ab177\Desktop\waterfall_plot_merge.png"
)