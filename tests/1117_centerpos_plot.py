import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_centers_by_row(parquet_file_path: str):
    """
    Plot x_center and y_center values by row number for an XPS group.
    
    Args:
        parquet_file_path: Path to the parquet file
    """
    # Load the data
    df = pd.read_parquet(parquet_file_path)
    
    # Create row numbers
    row_numbers = np.arange(len(df))
    
    # Plot setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot x_center vs row number
    ax1.plot(row_numbers, df['center_x'], 'bo-', markersize=3, linewidth=0.5, alpha=0.7)
    ax1.axhline(df['center_x'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["center_x"].mean():.2f}')
    ax1.axhline(df['center_x'].mean() + 3*df['center_x'].std(), color='orange', 
                linestyle=':', linewidth=1, label='±3σ')
    ax1.axhline(df['center_x'].mean() - 3*df['center_x'].std(), color='orange', 
                linestyle=':', linewidth=1)
    ax1.set_xlabel('Row Number (File Index)')
    ax1.set_ylabel('X Center (pixels)')
    ax1.set_title(f'X Center vs File Index\n{Path(parquet_file_path).name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot y_center vs row number
    ax2.plot(row_numbers, df['center_y'], 'go-', markersize=3, linewidth=0.5, alpha=0.7)
    ax2.axhline(df['center_y'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["center_y"].mean():.2f}')
    ax2.axhline(df['center_y'].mean() + 3*df['center_y'].std(), color='orange',
                linestyle=':', linewidth=1, label='±3σ')
    ax2.axhline(df['center_y'].mean() - 3*df['center_y'].std(), color='orange',
                linestyle=':', linewidth=1)
    ax2.set_xlabel('Row Number (File Index)')
    ax2.set_ylabel('Y Center (pixels)')
    ax2.set_title(f'Y Center vs File Index\n{Path(parquet_file_path).name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(parquet_file_path).parent / f"centers_by_row_{Path(parquet_file_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    print(f"Total files: {len(df)}")
    print(f"X center - Mean: {df['center_x'].mean():.3f}, Std: {df['center_x'].std():.3f}")
    print(f"Y center - Mean: {df['center_y'].mean():.3f}, Std: {df['center_y'].std():.3f}")
    print(f"X center range: [{df['center_x'].min():.3f}, {df['center_x'].max():.3f}]")
    print(f"Y center range: [{df['center_y'].min():.3f}, {df['center_y'].max():.3f}]")

# Alternative version with scatter plot and histograms
def plot_centers_comprehensive(parquet_file_path: str):
    """
    Comprehensive plot of centers with scatter plot and histograms.
    """
    # Load the data
    df = pd.read_parquet(parquet_file_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Scatter plot of centers
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.scatter(df['center_x'], df['center_y'], alpha=0.6, s=20)
    ax1.axhline(df['center_y'].mean(), color='red', linestyle='--', alpha=0.7)
    ax1.axvline(df['center_x'].mean(), color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('X Center (pixels)')
    ax1.set_ylabel('Y Center (pixels)')
    ax1.set_title('Center Positions Scatter Plot')
    ax1.grid(True, alpha=0.3)
    
    # X center vs row number
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(range(len(df)), df['center_x'], 'bo-', markersize=2, linewidth=0.5, alpha=0.7)
    ax2.axhline(df['center_x'].mean(), color='red', linestyle='--', label='Mean')
    ax2.set_xlabel('Row Number')
    ax2.set_ylabel('X Center (pixels)')
    ax2.set_title('X Center vs File Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Y center vs row number
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax3.plot(range(len(df)), df['center_y'], 'go-', markersize=2, linewidth=0.5, alpha=0.7)
    ax3.axhline(df['center_y'].mean(), color='red', linestyle='--', label='Mean')
    ax3.set_xlabel('Row Number')
    ax3.set_ylabel('Y Center (pixels)')
    ax3.set_title('Y Center vs File Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Histogram of center distances from mean
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    distances = np.sqrt((df['center_x'] - df['center_x'].mean())**2 + 
                       (df['center_y'] - df['center_y'].mean())**2)
    ax4.hist(distances, bins=50, alpha=0.7, color='purple')
    ax4.axvline(distances.mean(), color='red', linestyle='--', 
                label=f'Mean distance: {distances.mean():.3f}')
    ax4.set_xlabel('Distance from Mean Center (pixels)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Center Position Stability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(parquet_file_path).parent / f"centers_comprehensive_{Path(parquet_file_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive plot saved to: {output_path}")
    print(f"Mean center position: ({df['center_x'].mean():.3f}, {df['center_y'].mean():.3f})")
    print(f"Center position std: ({df['center_x'].std():.3f}, {df['center_y'].std():.3f})")
    print(f"Average distance from mean: {distances.mean():.3f} pixels")

# Usage examples:
if __name__ == "__main__":
    file_path = r"C:\Users\86177\Desktop\0809\analysis_parallel_time\xps_189.10625/filtered_xps_189.10625.parquet"
    
    # Simple version
    plot_centers_by_row(file_path)
    
    # Comprehensive version
    # plot_centers_comprehensive(file_path)