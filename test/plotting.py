import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the parquet file
df = pd.read_parquet(r'C:\Users\ab177\Desktop\diffraction_results\1005night\analysis_parallel_time\xps_172.36000\filtered_xps_172.36000.parquet')

# Select the radial_bin columns you specified
radial_columns = [f'radial_bin_{i:03d}' for i in range(50, 400, 30)]

# Filter only columns that exist in the dataframe
available_columns = [col for col in radial_columns if col in df.columns]
print(f"Found columns: {available_columns}")

# Select only the specified columns and drop rows with NaN in any of them
df_selected = df[available_columns].dropna()

print(f"Original rows: {len(df)}, After dropping NaN: {len(df_selected)}")
print(f"Removed {len(df) - len(df_selected)} rows with NaN values")

# Calculate correlation matrix
correlation_matrix = df_selected.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))

# Create heatmap with annotations
sns.heatmap(
    correlation_matrix,
    annot=True,            # Show correlation values in cells
    fmt='.2f',             # Format to 2 decimal places
    cmap='coolwarm',       # Color map: blue (-1) to red (+1)
    center=0,              # Center color map at 0
    square=True,           # Make cells square
    linewidths=0.5,        # Add lines between cells
    cbar_kws={'shrink': 0.8}  # Adjust color bar size
)

# Add title and adjust layout
plt.title('Correlation Heatmap of Radial Bin Columns', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot
plt.show()

# # Optional: Print summary statistics
# print("\nCorrelation Summary:")
# print("=" * 50)
# print(f"Number of columns analyzed: {len(available_columns)}")
# print(f"Shape of correlation matrix: {correlation_matrix.shape}")
# print(f"\nHighest correlation (excluding diagonal):")
# max_corr = correlation_matrix.where(~pd.DataFrame(np.eye(correlation_matrix.shape[0], dtype=bool), 
#                                                    index=correlation_matrix.index, 
#                                                    columns=correlation_matrix.columns)).max().max()
# min_corr = correlation_matrix.min().min()
# print(f"Maximum: {max_corr:.3f}")
# print(f"Minimum: {min_corr:.3f}")

# # Optional: Display the full correlation matrix as a table
# print("\nFull Correlation Matrix:")
# print(correlation_matrix.round(3))