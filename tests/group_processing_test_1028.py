import sys
import os

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.io.xps_value_sort import group_tiff_files_with_info, merge_xps_groups_strategy
from scripts.XPSGroup_process import BatchXPSProcessor
# 使用示例
def main():
    # 1. 分组文件
    grouped_files = merge_xps_groups_strategy(group_tiff_files_with_info(r"E:\20250808\8_water_IR72deg_longscan5\fist_AndorEMCCD"), tolerance = 0.002)
    
    # 2. 创建批量处理器
    batch_processor = BatchXPSProcessor(
        grouped_files=grouped_files,
        bkg_directory=r"C:\Users\86177\Desktop\Diffraction_code\Temp_savespace",
        bkg_filename="calc_bkg.tiff",
        output_directory=r"C:\Users\86177\Desktop\Diffraction_code\Temp_result_dir",
        initial_center_guess=(720, 350)
    )
    
    # 3. 处理所有组
    results = batch_processor.process_all_groups(
        radius=300,
        num_bins=300,
        parallel=False
    )
    
    # 4. 保存摘要和绘图
    batch_processor.save_summary()
    batch_processor.plot_all_profiles()
    
    print("Processing completed!")

if __name__ == "__main__":
    main()