import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the parquet file
df = pd.read_parquet(r'C:\Users\ab177\Desktop\diffraction_results\1005night\analysis_parallel_time\xps_172.40461\normalized_xps_172.40461.parquet')

# 1. 定义要绘制的radial_bin列名列表
target_cols = ["radial_bin_050", "radial_bin_080", "radial_bin_120", 
               "radial_bin_200", "radial_bin_350", "radial_bin_500"]

# 2. 创建2行3列的子图网格，设置画布大小
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=False, sharey=False)
# 将axes展平为一维数组（方便循环）
axes = axes.flatten()

# 3. 设置seaborn风格（可选，提升美观度）
sns.set_style("whitegrid")

# 4. 循环遍历每个列和对应的子图
for idx, col in enumerate(target_cols):
    ax = axes[idx]  # 获取当前子图
    
    # 绘制该列的分布直方图（带核密度曲线）
    sns.histplot(
        data=df,
        x=col,
        bins=50,  # 自定义分箱数，可根据数据调整
        kde=True,  # 显示核密度曲线
        color=sns.color_palette("tab20")[idx],  # 循环使用tab20配色，区分不同列
        edgecolor="black",  # 柱子边框，提升清晰度
        ax=ax  # 指定绘制在当前子图
    )
    
    # 为当前子图添加标题和标签
    ax.set_title(f"Distribution of {col}", fontsize=12, pad=8)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    # 调整刻度标签大小
    ax.tick_params(axis='both', labelsize=8)

# 5. 处理多余的子图（如果列数不足，此处6列刚好填满2x3，可省略）
# for idx in range(len(target_cols), len(axes)):
#     fig.delaxes(axes[idx])

# 6. 调整子图间距，避免标题/标签重叠
plt.tight_layout()

# 7. 添加总标题（可选）
fig.suptitle("Distribution of Different Radial Bin Columns", fontsize=16, y=1.02)

# 8. 显示图形
plt.show()