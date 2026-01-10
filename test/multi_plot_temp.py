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
        corrected_intensity = data[1] - avg_bkg
        corrected_data.append((data[0], corrected_intensity, data[2]))
    merge_data.extend(corrected_data)
#if the data[0] is closer than 0.01, we consider them as the same group and merge them
sorted_data = sorted(merge_data, key=lambda x: x[0])
final_data = []
tolerance = 0.01
i = 0
while i < len(sorted_data):
    current_xps = sorted_data[i][0]
    merged_intensity = sorted_data[i][1].copy()
    radial_distance = sorted_data[i][2]
    count = 1
    j = i + 1
    while j < len(sorted_data) and abs(sorted_data[j][0] - current_xps) < tolerance:
        merged_intensity += sorted_data[j][1]
        count += 1
        j += 1
    merged_intensity /= count
    final_data.append((current_xps, merged_intensity, radial_distance))
    i = j
print(f"Total groups after merging: {len(final_data)}")

# Execute the function with the mock data
plot_waterfall_diffraction(
    data=final_data, 
    scale=0.1, 
    width_to_height_ratio=0.4,
    filename=r"C:\Users\ab177\Desktop\waterfall_plot_merge1.png"
)