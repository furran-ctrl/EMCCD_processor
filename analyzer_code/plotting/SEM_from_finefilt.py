import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_and_plot_sem(parquet_file_path: str, save_plot_path: str = None, 
                          radial_bin_start: int = 40, radial_bin_end: int = 200,
                          figsize: tuple = (10, 6), plot_enabled: bool = True):
    """
    Calculate normalized SEM for a finefilt parquet file
    
    Parameters:
    - parquet_file_path: Path to the finefilt_xps_{}.parquet file
    - save_plot_path: Where to save the plot (optional)
    - radial_bin_start: Start bin for normalization (default: 40)
    - radial_bin_end: End bin for normalization (default: 200)
    - figsize: Figure size for the plot
    - plot_enabled: Whether to generate and show plots
    """
    
    # Load data
    df = pd.read_parquet(parquet_file_path)

    # Filter out images where radial_bin_000 is more than 3σ away from mean
    if 'radial_bin_000' in df.columns:
        bin_000_values = df['radial_bin_000'].values
        mean_bin_000 = np.mean(bin_000_values)
        std_bin_000 = np.std(bin_000_values)
        
        # Create filter mask: keep values within ±3σ
        bin_000_mask = (bin_000_values >= mean_bin_000 - 2 * std_bin_000) & \
                       (bin_000_values <= mean_bin_000 + 2 * std_bin_000)
        
        df_filtered = df[bin_000_mask].copy()
        
        print(f"Radial bin 000 filtering: {len(df)} → {len(df_filtered)} images "
              f"(removed {len(df) - len(df_filtered)} outliers)")
    else:
        df_filtered = df
        print("Warning: radial_bin_000 not found, skipping initial filtering")
    
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
    radial_data = df_filtered[radial_columns].values  # Shape: (n_images, 512)
    
    # Calculate normalization factors (sum from specified bins)
    normalization_factors = np.sum(radial_data[:, radial_bin_start:radial_bin_end+1], axis=1)
    
    # Calculate average intensity in normalization region
    avg_normalization_intensity = np.mean(normalization_factors)
    
    # Normalize each row by its normalization factor and multiply by average
    # This keeps the overall intensity scale roughly the same
    normalized_data = radial_data / normalization_factors[:, np.newaxis] * avg_normalization_intensity
    
    # Calculate statistics
    radial_avgs = np.mean(normalized_data, axis=0)
    radial_stds = np.std(normalized_data, axis=0)
    n_images = len(df_filtered)
    
    # Calculate SEM: std / avg / sqrt(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        sem_values = radial_stds / radial_avgs / np.sqrt(n_images)
        sem_values = np.nan_to_num(sem_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert radial bins to q values (0.024q = 1 pixel)
    q_values = np.arange(512) * 0.024
    
    # Create plot only if enabled
    if plot_enabled:
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
        'radial_avgs': radial_avgs,  # This is the key output for batch processing
        'normalization_factors': normalization_factors,
        'radial_stds': radial_stds,
        'n_images': n_images,
        'xps_value': xps_value
    }
    
    return results

def batch_process_sem(analysis_directory: str, output_dir: str = "./sem_analysis"):
    """
    Batch process all finefilt parquet files and generate combined CSV with radial averages
    No plotting during batch processing
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
    combined_data = {}
    
    print(f"Found {len(finefilt_files)} finefilt files. Processing without plotting...")
    
    for file_path in finefilt_files:
        try:
            # Extract XPS value from filename
            filename = file_path.name
            if 'finefilt_xps_' in filename:
                xps_str = filename[13:-8]
                try:
                    xps_value = float(xps_str)
                except ValueError:
                    print(f"Could not parse XPS value from {filename}, skipping")
                    continue
            else:
                continue
            
            print(f"Processing: {filename} (XPS = {xps_value:.5f})")
            
            # Calculate SEM without plotting
            results = calculate_and_plot_sem(
                str(file_path),
                plot_enabled=False  # Disable plotting during batch processing
            )
            
            # Store results
            all_results[filename] = results
            
            # Store radial averages for combined CSV
            xps_column_name = f"xps_{xps_value:.5f}"
            combined_data[xps_column_name] = results['radial_avgs']
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create combined DataFrame
    if combined_data:
        # Use q_values from the first result (all should be the same)
        first_key = list(all_results.keys())[0]
        q_values = all_results[first_key]['q_values']
        
        combined_df = pd.DataFrame({'q_values': q_values})
        
        # Add each XPS column
        for xps_column, radial_avgs in combined_data.items():
            combined_df[xps_column] = radial_avgs
        
        # Save combined CSV
        combined_path = output_dir / "combined_radial_averages.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined radial averages to {combined_path}")
        
        # Print summary
        print(f"\nBatch processing completed!")
        print(f"Processed {len(combined_data)} XPS groups")
        print(f"Output columns: q_values + {list(combined_data.keys())}")
        
    return all_results, combined_df

# Convenience function for single file with plotting
def analyze_single_file_with_plot(parquet_file_path: str, save_plot_path: str = None):
    """
    Convenience function to analyze a single file with plotting enabled
    """
    return calculate_and_plot_sem(
        parquet_file_path, 
        save_plot_path=save_plot_path,
        plot_enabled=True
    )

# Single file analysis WITH plotting
# results = analyze_single_file_with_plot(
#     r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking\analysis_parallel_time\xps_221.30750/finefilt_xps_221.30750.parquet",
#     save_plot_path=r"C:\Users\86177\Desktop\diffffraction/sem_plot_188.94000.png"
# )

# Batch processing WITHOUT plotting (for combined CSV)
all_results, combined_df = batch_process_sem(
    analysis_directory=r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking\analysis_parallel_time",
    output_dir=r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking"
)

# The combined CSV will have structure:
# q_values, xps_188.94000, xps_189.12000, xps_189.30000, ...

# Access results for a specific file
#q_values = all_results["finefilt_xps_188.94000.parquet"]['q_values']
#sem_values = all_results["finefilt_xps_188.94000.parquet"]['sem_values']
