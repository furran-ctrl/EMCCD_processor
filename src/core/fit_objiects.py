from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.core.reslut_class import ProcessedResult, XPSGroupResult

class XPSGroupProcessor:
    """
    处理单个XPS组中的所有文件
    """
    
    def __init__(self, 
                 xps_value: float,
                 file_list: List[str],
                 bkg_directory: str,
                 bkg_filename: str,
                 initial_center_guess: Tuple[float, float] = (512, 512)):
        """
        初始化XPS组处理器
        
        Args:
            xps_value: 该组的XPS值
            file_list: 该组的所有文件路径列表
            bkg_directory: 背景图像目录
            bkg_filename: 背景图像文件名
            initial_center_guess: 初始中心猜测坐标
        """
        self.xps_value = xps_value
        self.file_list = file_list
        self.bkg_directory = bkg_directory
        self.bkg_filename = bkg_filename
        self.initial_center_guess = initial_center_guess
        self.results: List[ProcessedResult] = []
        self.bkg_image: Optional[EMCCDimage] = None
        
    def load_background(self) -> None:
        """加载背景图像"""
        print(f"Loading background: {self.bkg_filename}")
        self.bkg_image = EMCCDimage(TiffLoader(self.bkg_directory, self.bkg_filename))
        
    def process_single_file(self, filepath: str) -> Optional[ProcessedResult]:
        """
        处理单个文件
        
        Returns:
            ProcessedResult or None if processing failed
        """
        try:
            # 加载图像
            image_file = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))
            
            # 扣除背景
            image_file.remove_background(self.bkg_image.get_processed_data())
            
            # 寻找中心
            center = image_file.iterative_ring_centroid(self.initial_center_guess)
            
            # 计算方位角平均
            bin_centers, radial_average = image_file.azimuthal_average()
            
            return ProcessedResult(
                filename=Path(filepath).name,
                xps_value=self.xps_value,
                center=center,
                radial_profile=radial_average,
                bin_centers=bin_centers
            )
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def process_group(self, 
                     radius: float = 300,
                     num_bins: int = 300,
                     verbose: bool = False) -> XPSGroupResult:
        """
        处理整个XPS组
        
        Args:
            radius: 方位角平均的最大半径
            num_bins: 径向分箱数量
            verbose: 是否显示详细进度
            
        Returns:
            XPSGroupResult: 组的平均结果
        """
        if self.bkg_image is None:
            self.load_background()
        
        print(f"Processing XPS group {self.xps_value:.5f} with {len(self.file_list)} files...")
        
        successful_results = []
        
        for i, filepath in enumerate(self.file_list):
            if verbose and i % 100 == 0:
                print(f"  Progress: {i}/{len(self.file_list)}")
            
            result = self.process_single_file(filepath)
            if result is not None:
                successful_results.append(result)
        
        # 计算组平均
        group_result = self.calculate_group_average(successful_results)
        
        print(f"Completed: {len(successful_results)}/{len(self.file_list)} files processed successfully")
        
        return group_result
    
    def calculate_group_average(self, results: List[ProcessedResult]) -> XPSGroupResult:
        """
        计算组的平均径向剖面
        
        Args:
            results: 成功的处理结果列表
            
        Returns:
            XPSGroupResult: 组的平均结果
        """
        if not results:
            raise ValueError("No successful results to average")
        
        # 确保所有结果的bin_centers相同
        reference_bins = results[0].bin_centers
        for result in results[1:]:
            if not np.allclose(result.bin_centers, reference_bins):
                raise ValueError("Inconsistent bin centers across files")
        
        # 堆叠所有径向剖面
        all_profiles = np.array([result.radial_profile for result in results])
        
        # 计算平均值和标准差
        avg_profile = np.mean(all_profiles, axis=0)
        std_profile = np.std(all_profiles, axis=0)
        
        # 收集所有中心坐标
        centers = [result.center for result in results]
        
        return XPSGroupResult(
            xps_value=self.xps_value,
            bin_centers=reference_bins,
            avg_radial_profile=avg_profile,
            std_radial_profile=std_profile,
            file_count=len(results),
            centers=centers
        )
