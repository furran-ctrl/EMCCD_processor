from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

from src.core.fit_objects import XPSGroupProcessor
from src.core.reslut_class import XPSGroupResult
from src.io.xps_value_sort import group_tiff_files_with_info, merge_xps_groups_strategy

class BatchXPSProcessor:
    """
    Batch processor for handling multiple XPS groups.
    
    This class organizes TIFF files into XPS groups, merges similar groups,
    and processes them sequentially using XPSGroupProcessor.
    """
    
    def __init__(self, 
                 data_directory: str,
                 bkg_directory: str,
                 bkg_filename: str,
                 azimuthal_radius: int,
                 initial_center_guess: Tuple[float, float] = (512, 512)):
        """
        Initialize batch XPS processor.
        
        Args:
            data_directory: Directory containing TIFF files to process
            bkg_directory: Directory containing background image
            bkg_filename: Background image filename
            initial_center_guess: Initial guess for diffraction center (x, y)
        """
        self.data_directory = data_directory
        self.bkg_directory = bkg_directory
        self.bkg_filename = bkg_filename
        self.azimuthal_radius = azimuthal_radius
        self.initial_center_guess = initial_center_guess
        self.merged_groups: List[Tuple[float, List[str]]] = []
        self.processed_results: List[XPSGroupResult] = []
    
    def organize_files_into_groups(self, threshold: int = 350, tolerance: float = 0.002) -> None:
        """
        Organize TIFF files into XPS groups and merge similar groups.
        
        Args:
            threshold: File count threshold for considering a group as 'large'
            tolerance: XPS difference tolerance for merging groups
        """
        print("Organizing files into XPS groups...")
        
        # Group files by XPS value
        groups_with_xps = group_tiff_files_with_info(self.data_directory)
        
        # Merge similar XPS groups
        self.merged_groups = merge_xps_groups_strategy(
            groups_with_xps, 
            threshold=threshold, 
            tolerance=tolerance
        )
        
        # Display group information
        print(f"\nXPS Groups Summary:")
        print("=" * 40)
        for xps_value, files in self.merged_groups:
            print(f"XPS {xps_value:.5f}: {len(files)} files")
        
        print(f"\nTotal: {len(self.merged_groups)} XPS groups")
    
    def process_groups_sequential(self, process_first_n: int = 2) -> List[XPSGroupResult]:
        """
        Process XPS groups sequentially (for testing purposes).
        
        Args:
            process_first_n: Number of groups to process (default: 2 for testing)
            
        Returns:
            List of XPSGroupResult objects for processed groups
        """
        if not self.merged_groups:
            raise ValueError("No groups available. Call organize_files_into_groups() first.")
        
        print(f"\nStarting sequential processing of first {process_first_n} XPS groups...")
        print("=" * 50)
        
        self.processed_results = []
        groups_to_process = self.merged_groups[:process_first_n]
        
        for i, (xps_value, file_list) in enumerate(groups_to_process):
            print(f"\nProcessing group {i+1}/{len(groups_to_process)}: XPS {xps_value:.5f}")
            print("-" * 40)
            
            try:
                # Create processor for current XPS group
                processor = XPSGroupProcessor(
                    xps_value=xps_value,
                    file_list=file_list,
                    bkg_directory=self.bkg_directory,
                    bkg_filename=self.bkg_filename,
                    azimuthal_radius=self.azimuthal_radius,
                    initial_center_guess=self.initial_center_guess
                )
                
                # Process the group with verbose output
                #group_result = processor.process_xps_group(verbose=True)
                #self.processed_results.append(group_result)
                processor.process_xps_group(verbose=True)
                processor.save_to_csv(r'C:\Users\86177\Desktop\Diffraction_code\Temp_result_dir')

                print(f"✓ Successfully processed XPS group {xps_value:.5f}")
                
            except Exception as e:
                print(f"✗ Failed to process XPS group {xps_value:.5f}: {e}")
                continue
        
        print(f"\nProcessing completed: {len(self.processed_results)}/{len(groups_to_process)} groups successful")
        return self.processed_results
    
    def get_processing_summary(self) -> dict:
        """
        Get summary of batch processing results.
        
        Returns:
            Dictionary containing batch processing summary
        """
        summary = {
            "total_groups": len(self.merged_groups),
            "processed_groups": len(self.processed_results),
            "total_files": sum(len(files) for _, files in self.merged_groups),
            "processed_files": sum(result.file_count for result in self.processed_results),
            "group_details": []
        }
        
        for result in self.processed_results:
            summary["group_details"].append({
                "xps_value": result.xps_value,
                "files_processed": result.file_count,
                "total_counts_mean": np.mean(result.total_counts) if result.total_counts else 0,
                "avg_profile_max": np.max(result.avg_radial_profile) if result.avg_radial_profile.size > 0 else 0
            })
        
        return summary