import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from processor_code.io.quick_plot import plot_ndarray
from processor_code.io.tiff_import import TiffLoader
import pandas as pd
import numpy as np

def plot_averaged_tiffs(parquet_path, row_indices, tiff_directory):
    """
    Read filenames from a .parquet file, load multiple TIFF files,
    calculate their mean, and plot the averaged array.
    
    Parameters:
    parquet_path (str): Path to the .parquet file containing filenames
    row_indices (list): List of row indices to read from the parquet file
    tiff_directory (str): Directory where TIFF files are located
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Get filenames from the specified rows (first column)
    filenames = df.iloc[row_indices, 0].tolist()
    
    # Load all TIFF files and store them in a list
    arrays = []
    for filename in filenames:
        array = TiffLoader(tiff_directory, filename)
        arrays.append(array)
    
    # Calculate the mean of all arrays
    averaged_array = np.mean(arrays, axis=0)

    # load background file
    bkg_array = TiffLoader(r"C:\Users\86177\Desktop\streaking","background.tiff")
    mean_array = averaged_array - bkg_array

    # Plot the averaged array
    plot_ndarray(mean_array, 50,2000)
    #param:[0,50]for bkg_removed, [450,700]for raw

# Example usage:
#plot_averaged_tiffs("path/to/your/file.parquet", [0, 1, 2, 3], r"D:\20250926\3_longscan5_44deg\fist_AndorEMCCD")
filelist = range(10,200)
parquetdir = r'C:\Users\86177\Desktop\streaking\analysis_parallel_time\xps_221.01956\normalized_xps_221.01956.parquet'
plot_averaged_tiffs(parquetdir, filelist, r"D:\20250926\3_longscan5_44deg\fist_AndorEMCCD")