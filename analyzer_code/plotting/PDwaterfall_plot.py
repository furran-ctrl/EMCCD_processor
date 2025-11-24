import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_pd_data(combined_csv_path: str, xps_t0: float):
    """
    Calculate PD data from combined CSV with XPS to ps conversion
    Returns: q_values, pd_data_dict, pd_times_ps, I0, ref_groups
    """
    # Load combined data
    df = pd.read_csv(combined_csv_path)
    
    # Extract q values and XPS columns
    q_values = df['q_values'].values
    xps_columns = [col for col in df.columns if col.startswith('xps_')]
    
    if len(xps_columns) < 3:
        raise ValueError("Need at least 3 XPS groups for PD calculation")
    
    # Extract XPS values from column names and get radial averages
    xps_data = []
    for col in xps_columns:
        xps_value = float(col[4:])  # Remove 'xps_' prefix
        radial_avgs = df[col].values
        xps_data.append((xps_value, radial_avgs))
    
    # Sort by XPS value (assuming XPS represents time)
    xps_data.sort(key=lambda x: x[0])
    xps_values = [item[0] for item in xps_data]
    radial_arrays = [item[1] for item in xps_data]
    
    # Convert XPS to ps: (xps_t0 - xps) / 0.1499 = t
    times_ps = [(xps_t0 - xps) / 0.1499 for xps in xps_values]
    
    # Find two largest XPS values for I0 reference
    sorted_by_xps = sorted(xps_data, key=lambda x: x[0], reverse=True)
    ref_group_1 = sorted_by_xps[0]  # (xps_value, radial_avgs)
    ref_group_2 = sorted_by_xps[1]  # (xps_value, radial_avgs)
    
    # Calculate I0 as average of two reference groups
    I0 = (ref_group_1[1] + ref_group_2[1]) / 2
    
    # Calculate PD for all groups except the two reference groups
    pd_data_dict = {}
    pd_times_ps = []
    pd_xps_values = []
    
    for (xps_value, radial_avgs), time_ps in zip(xps_data, times_ps):
        if xps_value not in [ref_group_1[0], ref_group_2[0]]:
            pd_values = (radial_avgs - I0) / I0 * 100
            pd_data_dict[xps_value] = pd_values
            pd_times_ps.append(time_ps)
            pd_xps_values.append(xps_value)
    
    # Sort PD data by time (ascending - earliest time first)
    sorted_indices = np.argsort(pd_times_ps)
    pd_times_ps_sorted = [pd_times_ps[i] for i in sorted_indices]
    pd_xps_values_sorted = [pd_xps_values[i] for i in sorted_indices]
    pd_data_sorted = {pd_xps_values_sorted[i]: pd_data_dict[pd_xps_values_sorted[i]] for i in sorted_indices}
    
    ref_groups = {
        'xps_1': ref_group_1[0],
        'xps_2': ref_group_2[0],
        'time_ps_1': (xps_t0 - ref_group_1[0]) / 0.1499,
        'time_ps_2': (xps_t0 - ref_group_2[0]) / 0.1499
    }
    
    return q_values, pd_data_sorted, pd_times_ps_sorted, I0, ref_groups

def plot_pd_waterfall(combined_csv_path: str, xps_t0: float, output_dir: str = None, STP: float = 2.0):
    """
    Plot PD waterfall plot from combined_radial_averages.csv
    with XPS to ps conversion: (xps_t0 - xps) / 0.1499 = t
    """
    
    # Calculate PD data
    q_values, pd_data, pd_times_ps, I0, ref_groups = calculate_pd_data(combined_csv_path, xps_t0)
    
    N = len(pd_data)
    
    if N == 0:
        raise ValueError("No PD data available for plotting")
    
    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(5, 0.6*N))
    
    ax2 = ax.twinx()
    
    s = q_values
    
    # Get sorted PD values and times
    pd_xps_values = list(pd_data.keys())
    pd_arrays = [pd_data[xps] for xps in pd_xps_values]
    
    # Plot each PD curve with vertical offset
    for i in range(N):
        ax.plot(s, pd_arrays[i] + i * STP, linewidth=1.5)
        ax.axhline(y=i * STP, linewidth='0.8', color='black', alpha=0.7)
    
    # Set up axes and labels
    ax.set_xlabel('q (Å⁻¹)', fontsize=15)
    ax.set_xticks(np.arange(0, np.round(s.max(), 0) + 1, 2))
    
    # Create y-axis labels with times in ps
    T = [0] * (N + 2)
    for i in range(N):
        T[i + 1] = pd_times_ps[i]
    
    left_labels = [f'{t:.2f}' for t in T]  # One decimal for ps
    
    ax.set_yticks(np.arange(-1, N + 1) * STP)
    ax.set_yticklabels(range(0,N+2), fontsize=12)
    ax2.set_yticks(np.arange(-1, N + 1) * STP)
    ax2.set_yticklabels(left_labels, fontsize=12)
    
    ax2.yaxis.tick_right()
    # Set x-ticks: 0, 2, 4, 6...
    q_max = np.round(s.max(), 0)
    ax.set_xticks(np.arange(0, q_max + 2, 2))
    
    # Set axis limits
    ax.set_ylim(-(STP - 0.01), STP * N - 0.01)
    ax2.set_ylim(-(STP - 0.01), STP * N - 0.01)
    
    ax.set_ylabel('PD = (I-I₀)/I₀ %', fontsize=15)
    ax2.set_ylabel('Time (ps)', fontsize=15)
    ax.grid(axis='x', alpha=0.3)
    
    plt.title(f'PD Waterfall Plot | I₀ = Average(XPS {ref_groups["xps_1"]:.2f}, XPS {ref_groups["xps_2"]:.2f})\n'
              f'Reference times: {ref_groups["time_ps_1"]:.1f} ps, {ref_groups["time_ps_2"]:.1f} ps', 
              fontsize=12)
    plt.tight_layout(pad = 20)
    
    # Save plot
    if output_dir is None:
        output_dir = Path(combined_csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_path = output_dir / "PD_waterfall_normal.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PD waterfall plot saved to: {output_path}")
    print(f"Reference groups: XPS {ref_groups['xps_1']:.2f} ({ref_groups['time_ps_1']:.1f} ps) "
          f"and XPS {ref_groups['xps_2']:.2f} ({ref_groups['time_ps_2']:.1f} ps)")
    print(f"PD calculated for {N} time points from {pd_times_ps[0]:.1f} ps to {pd_times_ps[-1]:.1f} ps")
    
    return pd_arrays, pd_times_ps, I0, ref_groups

def save_pd_data(combined_csv_path: str, xps_t0: float, output_dir: str = None):
    """
    Calculate and save PD data to CSV for further analysis
    with XPS to ps conversion
    """
    
    # Calculate PD data using the shared function
    q_values, pd_data, pd_times_ps, I0, ref_groups = calculate_pd_data(combined_csv_path, xps_t0)
    
    # Create DataFrame with PD data
    pd_data_dict = {'q_values': q_values}
    
    # Add PD columns with time labels
    for xps_value, time_ps in zip(pd_data.keys(), pd_times_ps):
        pd_data_dict[f'PD_{time_ps:.1f}ps'] = pd_data[xps_value]
    
    # Save PD data
    pd_df = pd.DataFrame(pd_data_dict)
    
    if output_dir is None:
        output_dir = Path(combined_csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    pd_output_path = output_dir / "PD_data.csv"
    pd_df.to_csv(pd_output_path, index=False)
    
    # Save reference information
    ref_info = {
        'parameter': ['xps_t0', 'reference_xps_1', 'reference_xps_2', 
                     'reference_time_ps_1', 'reference_time_ps_2'],
        'value': [xps_t0, ref_groups['xps_1'], ref_groups['xps_2'],
                 ref_groups['time_ps_1'], ref_groups['time_ps_2']]
    }
    ref_df = pd.DataFrame(ref_info)
    ref_output_path = output_dir / "PD_reference_info.csv"
    ref_df.to_csv(ref_output_path, index=False)
    
    print(f"PD data saved to: {pd_output_path}")
    print(f"Reference info saved to: {ref_output_path}")
    
    return pd_df

# Define your xps_t0 value (this should be provided)
xps_t0 = 221.0075  # Example value, replace with your actual xps_t0

# Generate PD waterfall plot with time in ps
PD_arrays, PD_times_ps, I0, ref_groups = plot_pd_waterfall(
    combined_csv_path=r"C:\Users\86177\Desktop\diffffraction\water-analyze\20250926streaking/combined_radial_averages.csv",
    xps_t0=xps_t0,
    output_dir=None,
    STP=20
)

# Save PD data with time in ps
# pd_df = save_pd_data(
#     combined_csv_path="./batch_sem_results/combined_radial_averages.csv",
#     xps_t0=xps_t0,
#     output_dir="./waterfall_plots"
# )