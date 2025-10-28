from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

from src.core.fit_objiects import XPSGroupProcessor
from src.core.reslut_class import XPSGroupResult

class BatchXPSProcessor:
    """
    批量处理所有XPS组
    """
    
    def __init__(self,
                 grouped_files: List[Tuple[float, List[str]]],
                 bkg_directory: str,
                 bkg_filename: str,
                 output_directory: str,
                 initial_center_guess: Tuple[float, float] = (512, 512)):
        """
        初始化批量处理器
        
        Args:
            grouped_files: 分组后的文件列表 [(xps_value, [file1, file2, ...]), ...]
            bkg_directory: 背景图像目录
            bkg_filename: 背景图像文件名
            output_directory: 结果输出目录
            initial_center_guess: 初始中心猜测坐标
        """
        self.grouped_files = grouped_files
        self.bkg_directory = bkg_directory
        self.bkg_filename = bkg_filename
        self.output_directory = Path(output_directory)
        self.initial_center_guess = initial_center_guess
        self.results: Dict[float, XPSGroupResult] = {}
        
        # 创建输出目录
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def process_all_groups(self,
                          radius: float = 300,
                          num_bins: int = 300,
                          parallel: bool = False,
                          max_workers: int = None) -> Dict[float, XPSGroupResult]:
        """
        处理所有XPS组
        
        Args:
            radius: 方位角平均半径
            num_bins: 径向分箱数量
            parallel: 是否并行处理
            max_workers: 并行工作进程数
            
        Returns:
            所有XPS组的结果字典
        """
        print(f"Processing {len(self.grouped_files)} XPS groups...")
        
        if parallel:
            return self._process_parallel(radius, num_bins, max_workers)
        else:
            return self._process_sequential(radius, num_bins)
    
    def _process_sequential(self, radius: float, num_bins: int) -> Dict[float, XPSGroupResult]:
        """顺序处理所有组"""
        for xps_value, file_list in self.grouped_files:
            print(f"\n{'='*50}")
            print(f"Processing XPS group: {xps_value:.5f}")
            print(f"{'='*50}")
            
            processor = XPSGroupProcessor(
                xps_value=xps_value,
                file_list=file_list,
                bkg_directory=self.bkg_directory,
                bkg_filename=self.bkg_filename,
                initial_center_guess=self.initial_center_guess
            )
            
            try:
                group_result = processor.process_group(radius, num_bins)
                self.results[xps_value] = group_result
                self._save_group_result(group_result)
                
            except Exception as e:
                print(f"Failed to process XPS group {xps_value}: {e}")
                continue
        
        return self.results
    
    def _process_parallel(self, radius: float, num_bins: int, max_workers: int = None) -> Dict[float, XPSGroupResult]:
        """并行处理所有组（需要额外实现）"""
        # 这里可以添加并行处理逻辑
        # 由于并行处理更复杂，这里先使用顺序处理
        print("Parallel processing not implemented yet, using sequential...")
        return self._process_sequential(radius, num_bins)
    
    def _save_group_result(self, group_result: XPSGroupResult) -> None:
        """保存单个组的结果"""
        # 保存为npy文件
        filename = f"xps_{group_result.xps_value:.5f}_result.npy"
        filepath = self.output_directory / filename
        
        # 保存所有数据
        save_data = {
            'xps_value': group_result.xps_value,
            'bin_centers': group_result.bin_centers,
            'avg_radial_profile': group_result.avg_radial_profile,
            'std_radial_profile': group_result.std_radial_profile,
            'file_count': group_result.file_count,
            'centers': group_result.centers
        }
        
        np.save(filepath, save_data)
        print(f"Saved results to: {filepath}")
    
    def save_summary(self) -> None:
        """保存处理摘要"""
        summary_file = self.output_directory / "processing_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("XPS Group Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for xps_value, result in self.results.items():
                f.write(f"XPS {xps_value:.5f}:\n")
                f.write(f"  Files processed: {result.file_count}\n")
                f.write(f"  Average center: {np.mean(result.centers, axis=0)}\n")
                f.write(f"  Center std: {np.std(result.centers, axis=0)}\n")
                f.write(f"  Max intensity: {np.max(result.avg_radial_profile):.2f}\n")
                f.write(f"  Saved to: xps_{xps_value:.5f}_result.npy\n\n")
        
        print(f"Summary saved to: {summary_file}")
    
    def plot_all_profiles(self) -> None:
        """绘制所有组的径向剖面"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            for xps_value, result in self.results.items():
                plt.plot(result.bin_centers, result.avg_radial_profile, 
                        label=f'XPS {xps_value:.5f}')
            
            plt.xlabel('Radial Distance (pixels)')
            plt.ylabel('Average Intensity')
            plt.title('Radial Profiles for All XPS Groups')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = self.output_directory / "all_profiles.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Plot saved to: {plot_file}")
            
        except ImportError:
            print("Matplotlib not available for plotting")