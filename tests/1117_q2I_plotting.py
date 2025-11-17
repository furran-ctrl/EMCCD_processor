import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_q2I_comparison(analyze_dir: str):
    """
    Calculate q^2I for all XPS groups with >1000 filtered data points and plot in one graph.
    
    Args:
        analyze_dir: Path to analysis directory (e.g., 'results/analyze_001')
    """
    analyze_path = Path(analyze_dir)
    
    # Find all filtered parquet files
    filtered_files = list(analyze_path.rglob("filtered_xps_*.parquet"))
    
    # Filter groups with >1000 data points
    valid_groups = []
    for file_path in filtered_files:
        df = pd.read_parquet(file_path)
        if len(df) >= 1000:
            # Extract XPS value from filename
            xps_value = float(file_path.stem.replace('filtered_xps_', ''))
            valid_groups.append((xps_value, file_path, df))
    
    if not valid_groups:
        print(f"No XPS groups found with >= 1000 filtered data points in {analyze_dir}")
        return
    
    print(f"Found {len(valid_groups)} XPS groups with >= 1000 filtered data points")
    
    # Plot setup
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_groups)))
    
    # Process each valid group
    for (xps_value, file_path, df), color in zip(valid_groups, colors):
        # Extract radial bin columns
        radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
        
        # Calculate average radial profile
        avg_radial_profile = df[radial_columns].mean()
        
        # Create bin indices and calculate q values
        bin_indices = np.array([int(col.split('_')[-1]) for col in radial_columns])
        q_values = bin_indices * 0.024  # q = bin_number * 0.024
        
        # Calculate q^2 * I for the average profile
        q2I = (q_values ** 2) * avg_radial_profile.values
        
        # Plot q^2I vs q
        plt.plot(q_values, q2I, color=color, linewidth=1.5, 
                label=f'XPS {xps_value:.2f} (n={len(df)})')
    
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('q²I (arb. units)')
    plt.title('q²I vs q for XPS Groups (≥1000 filtered data points)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = analyze_path / "q2I_comparison_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    print(f"Groups plotted: {[f'XPS {xps:.2f}' for xps, _, _ in valid_groups]}")

# Usage example:
if __name__ == "__main__":
    analyze_directory = r"C:\Users\86177\Desktop\0809\analysis_parallel_time"
    plot_q2I_comparison(analyze_directory)