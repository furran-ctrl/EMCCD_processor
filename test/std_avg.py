import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------------- 1. 模拟数据（替换为你的真实数据） ----------------------
# 模拟符合 Tuple[float, ndarray, ndarray, ndarray] 结构的测试数据
test_no: float = 1.0  # 测试编号（示例值）
x_values: np.ndarray = np.linspace(0, 20, 100)  # X轴数值
avg: np.ndarray = np.sin(x_values) + 5  # 平均值（示例曲线）
std: np.ndarray = np.random.rand(100) * 0.5  # 标准差（示例值）
data: Tuple[float, np.ndarray, np.ndarray, np.ndarray] = (test_no, avg, std, x_values)

# ---------------------- 2. 核心绘图逻辑 ----------------------
# 解包数据
test_no, avg_arr, std_arr, x_arr = data

# 创建画布（控制尺寸）
plt.figure(figsize=(10, 6))

# 绘制平均值实线
plt.plot(
    x_arr, avg_arr, 
    color='#2E86AB',  # 主色调（深蓝）
    linewidth=2,      # 线条宽度
    label='Average (test No.{})'.format(test_no)  # 图例标签
)

# 计算 ±3σ 边界
upper_bound = avg_arr + 3 * std_arr
lower_bound = avg_arr - 3 * std_arr

# 填充 ±3σ 半透明区域
plt.fill_between(
    x_arr,               # X轴范围
    lower_bound,         # 下界
    upper_bound,         # 上界
    color='#A23B72',     # 填充色（玫红）
    alpha=0.3,           # 透明度（0~1，0.3为半透明）
    label='±3σ (99.7% confidence)'  # 图例标签
)

# ---------------------- 3. 美化图表（可选但推荐） ----------------------
# 添加标题和坐标轴标签
plt.title('Average Value with ±3σ Confidence Band', fontsize=14, fontweight='bold')
plt.xlabel('X Values', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 添加图例（区分实线和填充区域）
plt.legend(loc='best', fontsize=10)

# 添加网格（增强可读性）
plt.grid(True, linestyle='--', alpha=0.5)

# 自动调整布局（避免标签重叠）
plt.tight_layout()

# ---------------------- 4. 显示/保存图表 ----------------------
plt.show()
# 如需保存图片，取消注释以下行
# plt.savefig('avg_with_3sigma.png', dpi=300, bbox_inches='tight')