import numpy as np
import matplotlib.pyplot as plt

def plot_waterfall_diffraction(data: list, xps_0: float, pixel_to_q: float, scale: float, width_to_height_ratio: float, filename: str = 'waterfall_plot.png'):
    """
    Generates a waterfall plot of Percentage Difference (PD) for diffraction data.

    PD is calculated as: PD = (I - I_0) / I_0 * 100%
    I_0 is the average of the two intensities with the largest XPS values.

    Args:
        data: list of [xps_value, [intensity values], [radial distance values]].
        xps_0: The reference XPS value for time calculation.
        pixel_to_q: Factor to convert radial distance to q (x-axis).
        scale: Vertical scaling factor for PD (controls vertical offset).
        width_to_height_ratio: Desired ratio of the plot width to height.
        filename: Name for the saved plot image file.
    """
    if not data:
        print("Error: The data list is empty.")
        return

    # 1. Sort data by XPS value (ascending) to ensure time-order plotting
    sorted_data = sorted(data, key=lambda x: x[0])

    # 2. Identify and calculate Background (I_0) from the two largest XPS groups
    if len(sorted_data) < 2:
        print("Error: Need at least two datasets to calculate background.")
        return

    bg_group_1 = sorted_data[-2]
    bg_group_2 = sorted_data[-1]

    I_bg_1 = np.array(bg_group_1[1])
    I_bg_2 = np.array(bg_group_2[1])

    # Calculate average background intensity (I_0)
    I_0 = (I_bg_1 + I_bg_2) / 2
    
    xps_bg_1 = bg_group_1[0]
    xps_bg_2 = bg_group_2[0]

    # 3. Prepare the figure and axes
    base_width = 8
    height = base_width / width_to_height_ratio
    
    fig, ax = plt.subplots(figsize=(base_width, height))
    time_tick_positions = []
    all_PD_values = []
    
    # Define a constant vertical step in %PD for visual separation
    vertical_step_PD = 1.0 * scale

    # 4. Process and Plot Each Group
    for i, group in enumerate(sorted_data):
        xps_value = group[0]
        I = np.array(group[1])
        radial_distance = np.array(group[2])

        # Calculate time (t)
        time_ps = (xps_value - xps_0) / 0.1499

        # Calculate PD: Percentage Difference
        with np.errstate(divide='ignore', invalid='ignore'):
            PD = np.divide(I - I_0, I_0, out=np.zeros_like(I_0, dtype=float), where=I_0!=0) * 100
        
        all_PD_values.extend(PD.tolist())
        
        # Calculate X-axis (q)
        q = radial_distance * pixel_to_q

        # The total offset for this curve
        vertical_offset = i * vertical_step_PD
        PD_curve = PD + vertical_offset
        
        # Plot the curve
        ax.plot(q, PD_curve, color=plt.cm.viridis(i/len(sorted_data)), linewidth=1)
        
        # Store the vertical position for the time label
        time_tick_positions.append(vertical_offset)


    # 5a. Set up the Right-Hand (Time) Axis
    ax_time = ax.twinx()
    ax_time.set_ylim(ax.get_ylim()) 

    # Create time labels
    time_labels = [f"{(group[0] - xps_0) / 0.1499:.3f} ps" for group in sorted_data]
    ax_time.set_yticks(time_tick_positions)
    ax_time.set_yticklabels(time_labels, fontsize=10, ha='left')
    ax_time.tick_params(axis='y', length=0)
    
    # 5b. Set up the Left-Hand (PD) Axis Ticks
    # Use the 'scale' to set the interval between the PD labels (e.g., scale=1.0 means 1% spacing)
    min_PD = np.min(all_PD_values)
    max_PD = np.max(all_PD_values)
    
    # Determine tick range for the base curve (i=0)
    min_tick = np.floor(min_PD / scale) * scale
    max_tick = np.ceil(max_PD / scale) * scale
    
    # Generate the base ticks and filter to only show relevant ticks for the bottom curve
    base_pd_ticks = np.arange(min_tick, max_tick + scale * 0.1, scale)
    visible_pd_ticks = base_pd_ticks[(base_pd_ticks >= min_PD - 0.1) & (base_pd_ticks <= max_PD + 0.5)]

    ax.set_yticks(visible_pd_ticks)
    pd_labels = [f"{t:.0f}\\%" for t in visible_pd_ticks]
    ax.set_yticklabels(pd_labels)
    
    # 6. Set Title
    title = (f"Waterfall Plot of Percentage Difference (PD)\n"
             f"Background $I_0$ averaged from $XPS$ at ${xps_bg_1}$ and ${xps_bg_2}$")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$q$ (Normalized Radial Distance)', fontsize=12)
    ax.set_ylabel('Percentage Difference ($\%PD$)', fontsize=12, loc='top')

    # 7. Final Touches (Grid lines for base PD level)
    ax.yaxis.grid(False) 
    for tick in visible_pd_ticks:
        ax.axhline(tick, color='gray', linestyle=':', linewidth=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt

def plot_waterfall_diffraction_final(data: list, xps_0: float, pixel_to_q: float, scale: float, width_to_height_ratio: float, filename: str = 'waterfall_plot_final.png'):
    """
    Generates the final customized waterfall plot of Percentage Difference (PD).

    Incorporates fixed width, flipped XPS order, aligned baselines, and specific 
    y-axis range calculation.

    Args:
        data: list of [xps_value, [intensity values], [radial distance values]].
        xps_0: The reference XPS value for time calculation.
        pixel_to_q: Factor to convert radial distance to q (x-axis).
        scale: Vertical scaling factor. Used for PD labels' step size (e.g., 1.0 = 1% step).
        width_to_height_ratio: Desired ratio of the plot width to height.
        filename: Name for the saved plot image file.
    """
    if not data or len(data) < 2:
        print("Error: The data list is empty or has fewer than two groups.")
        return

    # 1. Sort data by XPS value and Reverse Order (Biggest XPS at the bottom, i=0)
    # Sort descending by XPS, then reverse to have smallest XPS (earliest time) at i=0
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    # 2. Identify and calculate Background (I_0) from the two largest XPS groups (now at the end)
    bg_group_1 = sorted_data[0] 
    bg_group_2 = sorted_data[1]
    I_0 = (np.array(bg_group_1[1]) + np.array(bg_group_2[1])) / 2
    xps_bg_1, xps_bg_2 = bg_group_1[0], bg_group_2[0]

    # 3. Setup Figure Dimensions (Width fixed at 8)
    plot_width = 8
    plot_height = plot_width / width_to_height_ratio
    
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    time_tick_positions = []
    
    # Define the constant vertical step in %PD for visual separation (e.g., 4% PD separation)
    vertical_step_PD = 1.0 * scale 

    # 4. Process and Plot Each Group
    num_curves = len(sorted_data)
    for i, group in enumerate(sorted_data):
        xps_value = group[0]
        I = np.array(group[1])
        radial_distance = np.array(group[2])

        # Calculate PD
        with np.errstate(divide='ignore', invalid='ignore'):
            PD = np.divide(I - I_0, I_0, out=np.zeros_like(I_0, dtype=float), where=I_0!=0) * 100
        
        q = radial_distance * pixel_to_q
        
        # Vertical offset is proportional to the index 'i'
        vertical_offset = i * vertical_step_PD
        PD_curve = PD + vertical_offset
        
        # Plot the curve
        ax.plot(q, PD_curve, linewidth=1.5)
        
        # ALIGNMENT: Draw the horizontal baseline (PD=0) at the curve's offset position
        ax.axhline(vertical_offset, color='gray', linestyle='--', linewidth=0.8, zorder=0)

        time_tick_positions.append(vertical_offset)

    
    # 5. Set Y-Axis Limits (to leave room for one extra offset on top and bottom)
    
    # The total vertical range of the plotted data is:
    # Max Y: (Max index * vertical_step_PD) + Max positive PD deviation
    max_data_y = time_tick_positions[-1] 
    # Min Y: (Min index * vertical_step_PD) + Min negative PD deviation
    min_data_y = time_tick_positions[0]  # Since time_tick_positions[0] is 0
    
    # Add one extra vertical step to the top and bottom limits
    y_limit_top = max_data_y + vertical_step_PD
    y_limit_bottom = min_data_y - vertical_step_PD
    
    ax.set_ylim(y_limit_bottom, y_limit_top)

    
    # 6. Set up the Right-Hand (Time) Axis
    ax_time = ax.twinx()
    ax_time.set_ylim(ax.get_ylim()) 

    # Calculate time
    time_labels = [f"{(xps_0 - group[0]) / 0.1499:.3f} ps" for group in sorted_data]
    ax_time.set_yticks(time_tick_positions)
    ax_time.set_yticklabels(time_labels, fontsize=10, ha='left', va='center') # va='center' ensures perfect alignment with baseline
    ax_time.tick_params(axis='y', length=0)
    
    
    # 7. Set up the Left-Hand (PD) Axis Ticks
    
    # The PD labels should be placed at the baseline of each curve (vertical_offset)
    # The labels should be 0%, (N-1)*scale %, (N-1)*scale*2 % ...
    
    # Calculate the label value for each baseline (offset)
    # PD Label at i=0 is 0%.
    # PD Label at i=1 is vertical_step_PD * scale % (if scale is 1, it's 4%)
    # PD Label at i=2 is vertical_step_PD * 2 * scale % (if scale is 1, it's 8%)
    
    # NOTE: Since the curves are already shifted by vertical_step_PD, 
    # the labels should reflect the CUMULATIVE SHIFT.
    
    # PD labels for each baseline: 0%, 4%, 8%, ...
    # The `vertical_offset` array already represents the baseline positions (0, 4, 8, ...)
    pd_baseline_labels = [f"{v:.0f}\\%" for v in time_tick_positions]

    ax.set_yticks(time_tick_positions)
    ax.set_yticklabels(pd_baseline_labels)
    ax.tick_params(axis='y', which='major', length=5) 
    
    # 8. Set Labels and Title
    title = (f"Waterfall Plot of Percentage Difference (PD) - Aligned\n"
             f"Background $I_0$ averaged from $XPS$ at ${xps_bg_1}$ and ${xps_bg_2}$ (Reversed Order)")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$q$ (Normalized Radial Distance)', fontsize=12)
    ax.set_ylabel('Percentage Difference ($\%PD$)', fontsize=12, loc='top')
    
    ax.grid(False) # Turn off all default grid lines
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final aligned waterfall plot saved as '{filename}'")