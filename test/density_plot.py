import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # 用于计算密度
import pandas as pd

# Load the parquet file
df = pd.read_parquet(r'C:\Users\ab177\Desktop\diffraction_results\merged\water_scan\xps_172.35982\normalized_xps_172.35982.parquet')
# 设置绘图风格
sns.set_style("whitegrid")

# 1. 准备数据
x = df["radial_bin_400"]
y = df["radial_bin_080"]

# 2. 计算点的局部密度（核心：用高斯核估计密度）
from scipy.stats import gaussian_kde
# 过滤缺失值
mask = ~np.isnan(x) & ~np.isnan(y)
x_valid = x[mask]
y_valid = y[mask]
# 计算密度值
kde = gaussian_kde(np.vstack([x_valid, y_valid]))
density = kde(np.vstack([x_valid, y_valid]))

# 3. 绘制密度散点图
fig, ax = plt.subplots(figsize=(10, 8))
scatter = sns.scatterplot(
    x=x_valid,
    y=y_valid,
    c=density,  # 用密度值映射颜色
    cmap="viridis",  # 配色方案（推荐：viridis/plasma/inferno）
    s=20,  # 点的大小
    alpha=0.7,  # 透明度，避免重叠遮挡
    edgecolor=None,  # 关闭点的边框
    ax=ax
)

# 4. 添加颜色条（展示密度与颜色的对应关系）
cbar = plt.colorbar(scatter.collections[0], ax=ax)
cbar.set_label("Point Density", fontsize=10)

# 5. 添加标题和标签
ax.set_title("Density Scatter Plot: radial_bin_080 vs radial_bin_160", fontsize=14, pad=10)
ax.set_xlabel("radial_bin_080", fontsize=12)
ax.set_ylabel("radial_bin_160", fontsize=12)

# 6. 优化布局
plt.tight_layout()
plt.show()