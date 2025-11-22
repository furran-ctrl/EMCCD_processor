import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

class CenterCoordinateFilter:
    """
    Step 1: Filter data based on center_x and center_y coordinates
    Uses first XPS group to calculate std, then applies to all groups
    """
    
    def __init__(self, analysis_directory: str):
        self.analysis_dir = Path(analysis_directory)
        self.center_stats = {}
        
    def _remove_outliers_mad(self, data: np.ndarray, n_mads: float = 8.0) -> np.ndarray:
        """Remove outliers using Median Absolute Deviation"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        lower_bound = median - n_mads * mad
        upper_bound = median + n_mads * mad
            
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data[mask]
    
    def calculate_reference_std(self) -> Tuple[float, float]:
        """Calculate σ for center_x and center_y from first XPS group (with outlier removal)"""
        # Find all parquet files
        parquet_files = sorted(list(self.analysis_dir.glob("**/xps_*.parquet")))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.analysis_dir}")
        
        # Use first XPS group as reference
        first_file = parquet_files[0]
        print(f"Using first XPS group as reference: {first_file.name}")
        
        try:
            df = pd.read_parquet(first_file)
            
            # Remove outliers from center coordinates using MAD
            center_x_clean = self._remove_outliers_mad(df['center_x'].dropna().values)
            center_y_clean = self._remove_outliers_mad(df['center_y'].dropna().values)
            
            # Calculate std on cleaned data
            sigma_x = np.std(center_x_clean)
            sigma_y = np.std(center_y_clean)
            
            # Store statistics
            self.center_stats = {
                'sigma_x': sigma_x,
                'sigma_y': sigma_y,
                'mean_x_first_group': np.mean(center_x_clean),
                'mean_y_first_group': np.mean(center_y_clean),
                'reference_file': str(first_file),
                'original_count_first_group': len(df),
                'cleaned_count_first_group': len(center_x_clean)  # Should be same for x and y
            }
            
            print(f"Reference σ calculated from {first_file.name}:")
            print(f"  center_x: σ = {sigma_x:.3f} (from {len(center_x_clean)} cleaned points)")
            print(f"  center_y: σ = {sigma_y:.3f} (from {len(center_y_clean)} cleaned points)")
            
            return sigma_x, sigma_y
            
        except Exception as e:
            raise ValueError(f"Error processing reference file {first_file}: {e}")
    
    def filter_single_file(self, file_path: Path, sigma_x: float, sigma_y: float) -> pd.DataFrame:
        """Filter a single parquet file based on center coordinate criteria"""
        df = pd.read_parquet(file_path)
        
        # Remove outliers from center coordinates using MAD
        center_x_clean = self._remove_outliers_mad(df['center_x'].dropna().values)
        center_y_clean = self._remove_outliers_mad(df['center_y'].dropna().values)

        # Calculate file-specific means for center coordinates
        mean_x = np.mean(center_x_clean)
        mean_y = np.mean(center_y_clean)
        
        # Filter: remove points outside ±4σ (using reference sigma)
        mask = (
            (df['center_x'] >= mean_x - 4 * sigma_x) & 
            (df['center_x'] <= mean_x + 4 * sigma_x) &
            (df['center_y'] >= mean_y - 4 * sigma_y) & 
            (df['center_y'] <= mean_y + 4 * sigma_y)
        )
        
        filtered_df = df[mask].copy()
        return filtered_df
    
    def process_all_files(self) -> Dict[str, Dict]:
        """Process all XPS files using reference std from first group"""
        # Calculate reference std from first group
        sigma_x, sigma_y = self.calculate_reference_std()
        results = {}
        
        # Find all parquet files
        parquet_files = list(self.analysis_dir.glob("**/xps_*.parquet"))
        
        for file_path in parquet_files:
            try:
                filtered_df = self.filter_single_file(file_path, sigma_x, sigma_y)
                
                # Save filtered file
                output_filename = f"filtered_{file_path.name}"
                output_path = file_path.parent / output_filename
                filtered_df.to_parquet(output_path)
                
                # Store results
                original_count = len(pd.read_parquet(file_path))
                filtered_count = len(filtered_df)
                removed_count = original_count - filtered_count
                
                results[str(file_path)] = {
                    'original_count': original_count,
                    'filtered_count': filtered_count,
                    'removed_count': removed_count,
                    'removed_ratio': removed_count / original_count if original_count > 0 else 0,
                    'mean_x': filtered_df['center_x'].mean(),
                    'mean_y': filtered_df['center_y'].mean(),
                    'std_x': filtered_df['center_x'].std(),
                    'std_y': filtered_df['center_y'].std()
                }
                
                print(f"Processed {file_path.name}: {original_count} → {filtered_count} "
                      f"({removed_count} removed, {removed_count/original_count*100:.1f}%)")
                      
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[str(file_path)] = {'error': str(e)}
        
        # Add reference statistics to results
        results['_reference_stats'] = self.center_stats
        
        return results
    
    def get_reference_statistics(self) -> Dict:
        """Get the reference statistics used for filtering"""
        return self.center_stats.copy()