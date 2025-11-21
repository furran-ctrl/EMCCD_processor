import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_radial_bin_std(parquet_file_path: str):
    """
    Plot the standard deviation of radial bins for a processed parquet file.
    
    Args:
        parquet_file_path: Path to the parquet file
    """
    # Load the data
    df = pd.read_parquet(parquet_file_path)

    # Extract radial bin columns
    radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
    
    # Define the normalization range (bins 10 to 100)
    '''norm_columns = [f'radial_bin_{i:03d}' for i in range(30, 200)]
    
    # Calculate normalization factor for each file (sum of bins 10-100)
    #normalization_factors = df[norm_columns].sum(axis=1)
    # Calculate normalization factor for each file: sum(bin_number^2 * radial_bin) over bins 30-150
    normalization_factors = np.zeros(len(df))
    for i, col in enumerate(norm_columns):
        bin_num = 30 + i  # bin numbers from 30 to 149
        normalization_factors += df[col].values * (bin_num ** 1)
    
    # Normalize each radial profile by the sum of bins 10-100
    normalized_radial_data = df[radial_columns].div(normalization_factors, axis=0)'''
    normalized_radial_data = df[radial_columns]
    #calculate the log for each data in normalized_radial_data
    #normalized_radial_data = np.log1p(normalized_radial_data)
    
    # Calculate mean and standard deviation for each normalized radial bin
    radial_means = normalized_radial_data.mean()
    radial_means = np.log1p(radial_means)
    radial_stds = normalized_radial_data.std()
    
    # Calculate std/mean (coefficient of variation)
    radial_std_over_mean = radial_stds / radial_means
    #radial_std_over_mean /= len(df)**0.5
    
    # Create bin indices from column names
    bin_indices = [int(col.split('_')[-1])*0.024 for col in radial_columns]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(bin_indices, radial_means.values, 'b-', linewidth=2)
    plt.xlabel('Radial Bin Index')
    plt.ylabel('Standard Deviation')
    plt.title(f'Standard Deviation of Radial Bins\n{Path(parquet_file_path).name}')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = Path(parquet_file_path).parent / f"radial_std_{Path(parquet_file_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    print(f"Total files processed: {len(df)}")
    print(f"Mean std across all bins: {radial_stds.mean():.4f}")

# Usage example:
if __name__ == "__main__":
    file_path = r"C:\Users\86177\Desktop\0809\analysis_parallel_time\xps_189.69000/filtered_xps_189.69000.parquet"
    plot_radial_bin_std(file_path)