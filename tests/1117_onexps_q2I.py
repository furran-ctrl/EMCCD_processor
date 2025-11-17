import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_all_q2I_individual(parquet_file_path: str, max_plots: int = 100):
    """
    Plot q^2I for all individual files in one parquet file.
    
    Args:
        parquet_file_path: Path to the parquet file
        max_plots: Maximum number of individual curves to plot (for performance)
    """
    # Load the data
    df = pd.read_parquet(parquet_file_path)
    
    # Extract radial bin columns
    radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
    
    # Create bin indices and calculate q values
    bin_indices = np.array([int(col.split('_')[-1]) for col in radial_columns])
    q_values = bin_indices * 0.024  # q = bin_number * 0.024
    
    # Plot setup
    plt.figure(figsize=(12, 8))
    
    # Limit number of plots for performance
    n_plots = min(len(df), max_plots)
    
    # Plot each individual file
    for i in range(n_plots):
        # Get radial profile for this file
        radial_profile = df[radial_columns].iloc[i]
        
        # Calculate q^2 * I for this file
        q2I = (q_values ** 2) * radial_profile.values
        
        # Plot with transparency to see overlapping curves
        plt.plot(q_values, q2I, alpha=0.05, linewidth=0.8, color='blue')
    
    # Calculate and plot the average q²I
    avg_radial_profile = df[radial_columns].mean()
    avg_q2I = (q_values ** 2) * avg_radial_profile.values
    #plt.plot(q_values, avg_q2I, 'r-', linewidth=3, label='Average', color='red')
    
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('q²I (arb. units)')
    plt.title(f'Individual q²I Curves\n{Path(parquet_file_path).name} (n={n_plots} files)')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(parquet_file_path).parent / f"individual_q2I_curves_{Path(parquet_file_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    print(f"Plotted {n_plots} individual curves + average")
    print(f"Total files in dataset: {len(df)}")

# Usage examples:
if __name__ == "__main__":
    file_path = r"C:\Users\86177\Desktop\0809\analysis_parallel_time\xps_189.10625/filtered_xps_189.10625.parquet"
    
    # For small datasets (<100 files)
    plot_all_q2I_individual(file_path)