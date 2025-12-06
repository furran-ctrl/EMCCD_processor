import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the parquet file
df = pd.read_parquet(r'C:\Users\ab177\Desktop\diffraction_results\1004\analysis_parallel_time\xps_173.03500\filtered_xps_173.03500.parquet')

# Select the radial_bin columns you specified
radial_columns = [f'radial_bin_{i:03d}' for i in range(60, 71, 2)]

# Filter only columns that exist in the dataframe
available_columns = [col for col in radial_columns if col in df.columns]
print(f"Found columns: {available_columns}")

# Select only the specified columns and drop rows with NaN in any of them
df_selected = df[available_columns].dropna()

# Create the heatmap
plt.figure(figsize=(12, 10))

sns.boxplot(df_selected)

# Show the plot
plt.show()