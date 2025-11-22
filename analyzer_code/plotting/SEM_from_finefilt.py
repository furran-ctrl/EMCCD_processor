import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_and_plot_sem(parquet_file_path: str, save_plot_path: str = None, 
                          radial_bin_start: int = 40, radial_bin_end: int = 200,
                          figsize: tuple = (10, 6)):
    """
    Calculate and plot normalized SEM for a finefilt parquet file
    
    Parameters:
    - parquet_file_path: Path to the finefilt_xps_{}.parquet file
    - save_plot_path: Where to save the plot (optional)
    - radial_bin_start: Start bin for normalization (default: 40)
    - radial_bin_end: End bin for normalization (default: 200)
    - figsize: Figure size for the plot
    """
    
    # Load data
    df = pd.read_parquet(parquet_file_path)
    
    # Extract XPS value from filename for title
    file_path = Path(parquet_file_path)
    xps_value = None
    if 'finefilt_xps_' in file_path.name:
        try:
            xps_str = file_path.name[13:-8]  # Remove 'finefilt_xps_' and '.parquet'
            xps_value = float(xps_str)
        except ValueError:
            pass
    
    # Extract radial bin columns (000 to 511)
    radial_columns = [f'radial_bin_{i:03d}' for i in range(512)]
    radial_data = df[radial_columns].values  # Shape: (n_images, 512)
    
    # Calculate normalization factors (sum from specified bins)
    normalization_factors = np.sum(radial_data[:, radial_bin_start:radial_bin_end+1], axis=1)
    
    # Avoid division by zero
    normalization_factors[normalization_factors == 0] = 1.0
    
    # Normalize each row by its normalization factor
    normalized_data = radial_data / normalization_factors[:, np.newaxis]
    
    # Calculate statistics
    radial_avgs = np.mean(normalized_data, axis=0)
    radial_stds = np.std(normalized_data, axis=0)
    n_images = len(df)
    
    # Calculate SEM: std / avg / sqrt(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        sem_values = radial_stds / radial_avgs / np.sqrt(n_images)
        sem_values = np.nan_to_num(sem_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert radial bins to q values (0.024q = 1 pixel)
    q_values = np.arange(512) * 0.024
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot SEM vs q
    plt.plot(q_values, sem_values, 'b-', linewidth=1.5, label='Normalized SEM')
    
    # Customize plot
    plt.xlabel('q (Å⁻¹)', fontsize=12)
    plt.ylabel('Normalized SEM', fontsize=12)
    
    # Title
    if xps_value is not None:
        title = f'Normalized SEM vs q (XPS = {xps_value:.5f}, N = {n_images} images)'
    else:
        title = f'Normalized SEM vs q (N = {n_images} images)'
    plt.title(title, fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Set axis limits
    plt.xlim(0, q_values[-1])
    valid_sem = sem_values[np.isfinite(sem_values) & (sem_values > 0)]
    if len(valid_sem) > 0:
        y_max = np.max(valid_sem) * 1.1
        plt.ylim(0, y_max)
    
    # Add statistics text
    stats_text = f'Normalization: bins {radial_bin_start:03d}-{radial_bin_end:03d}\n'
    stats_text += f'Images: {n_images}\n'
    stats_text += f'Avg norm factor: {np.mean(normalization_factors):.2e}'
    
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved SEM plot to {save_plot_path}")
    
    plt.show()
    
    # Return results for further analysis
    results = {
        'q_values': q_values,
        'sem_values': sem_values,
        'normalization_factors': normalization_factors,
        'radial_avgs': radial_avgs,
        'radial_stds': radial_stds,
        'n_images': n_images,
        'xps_value': xps_value
    }
    
    return results

def batch_plot_sem(analysis_directory: str, output_dir: str = "./sem_plots"):
    """
    Batch process all finefilt parquet files in a directory and generate SEM plots
    """
    analysis_dir = Path(analysis_directory)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all finefilt parquet files
    finefilt_files = list(analysis_dir.glob("**/finefilt_xps_*.parquet"))
    
    if not finefilt_files:
        print(f"No finefilt parquet files found in {analysis_directory}")
        return {}
    
    all_results = {}
    
    for file_path in finefilt_files:
        try:
            print(f"Processing: {file_path.name}")
            
            # Create output plot path
            plot_filename = f"sem_{file_path.stem}.png"
            plot_path = output_dir / plot_filename
            
            # Calculate and plot SEM
            results = calculate_and_plot_sem(
                str(file_path),
                save_plot_path=str(plot_path)
            )
            
            all_results[file_path.name] = results
            print(f"  → Generated plot: {plot_filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save combined data to CSV
    if all_results:
        save_combined_sem_data(all_results, output_dir)
    
    print(f"\nProcessed {len(all_results)} files. Results saved to {output_dir}")
    return all_results

def save_combined_sem_data(all_results: dict, output_dir: Path):
    """Save combined SEM data from all files to CSV"""
    combined_data = []
    
    for filename, results in all_results.items():
        # Extract XPS value from filename
        xps_value = results['xps_value']
        
        for i, (q, sem) in enumerate(zip(results['q_values'], results['sem_values'])):
            combined_data.append({
                'filename': filename,
                'xps_value': xps_value if xps_value is not None else np.nan,
                'radial_bin_index': i,
                'q_value': q,
                'normalized_sem': sem,
                'n_images': results['n_images']
            })
    
    combined_df = pd.DataFrame(combined_data)
    combined_path = output_dir / "combined_sem_data.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined SEM data to {combined_path}")

# Single file
results = calculate_and_plot_sem(
    r"C:\Users\86177\Desktop\0809\analysis_parallel_time\xps_189.84000/finefilt_xps_189.84000.parquet",
    save_plot_path=r"C:\Users\86177\Desktop\temparquet"
)

'''# Batch process all files in a directory
all_results = batch_plot_sem(
    analysis_directory="results/analysis_20240115",
    output_dir="./my_sem_plots"
)

# Access results for a specific file
q_values = all_results["finefilt_xps_188.94000.parquet"]['q_values']
sem_values = all_results["finefilt_xps_188.94000.parquet"]['sem_values']'''
