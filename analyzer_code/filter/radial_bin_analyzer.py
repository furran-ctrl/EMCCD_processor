import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import Dict, List, Optional
import os

class RadialBinAnalyzer:
    """
    Process radial bins with baseline removal and statistical analysis
    """
    
    def __init__(self, export_base_dir: str = "./radial_analysis_results"):
        self.export_base_dir = Path(export_base_dir)
        self.export_base_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_baseline(self, intensities: np.ndarray, window_size: int = 50, sigma: float = 5.0) -> np.ndarray:
        """Calculate baseline using moving average + Gaussian smoothing"""
        if len(intensities) < window_size:
            window_size = len(intensities) // 2
            if window_size < 2:
                return intensities.copy()
        
        # Simple moving average
        baseline = np.convolve(intensities, np.ones(window_size)/window_size, mode='same')
        
        # Handle edges
        half_window = window_size // 2
        if half_window > 0:
            baseline[:half_window] = np.mean(intensities[:window_size])
            baseline[-half_window:] = np.mean(intensities[-window_size:])
        
        # Gaussian smoothing
        baseline_smoothed = gaussian_filter1d(baseline, sigma=sigma)
        
        return baseline_smoothed
    
    def fine_analysis(self, intensities: np.ndarray) -> Dict:
        """
        Perform fine analysis on radial bin data
        Returns: sigma_ideal, sigma_shift, sigma_exp
        """
        # Calculate baseline
        baseline = self.calculate_baseline(intensities)

        # Remove baseline
        baseline_removed = intensities - baseline
        # nan_indices = [i for i, x in enumerate(baseline_removed) if np.isnan(x)]
        # print(f"NaN indices: {nan_indices}")

        # Calculate mad_filter (mad of baseline-removed data)
        baseline_median = np.median(baseline_removed)
        mad_filter = np.median(np.abs(baseline_removed - baseline_median))
        
        #Case check to save the processor in case of nan from calculate_baseline
        if np.isnan(mad_filter):
            # Return empty results with no filtering
            return {
                'sigma_ideal': np.nan,
                'sigma_shift': np.nan,
                'sigma_exp': np.nan,
                'avg_intensity': np.nan,
                'baseline': np.array([]),
                'baseline_removed': np.array([]),
                'filtered_data': np.array([]),
                'filtered_baseline_removed': np.array([]),
                'outlier_mask': np.ones(len(intensities), dtype=bool),
                'is_outlier': np.zeros(len(intensities), dtype=bool),
                'original_count': len(intensities),
                'filtered_count': 0,
                'removal_ratio': 0.0
            }

        # Filter outliers: remove data where |data - baseline| > 4 * MAD
        outlier_mask = np.abs(baseline_removed) <= 4 * mad_filter
        filtered_data = intensities[outlier_mask]
        filtered_baseline_removed = baseline_removed[outlier_mask]
        
        # Calculate sigma_exp (std of original data)
        sigma_exp = np.std(filtered_data)

        # Calculate sigma_ideal (std of filtered baseline-removed data) and sigme_shift
        sigma_ideal = np.std(filtered_baseline_removed) if len(filtered_baseline_removed) > 0 else 0.0
        if sigma_exp**2-sigma_ideal**2 > 0:
            sigma_shift = (sigma_exp**2-sigma_ideal**2)**0.5
        else: 
            sigma_shift = 0 

        # Calculate average of filtered data
        avg_intensity = np.mean(filtered_data) if len(filtered_data) > 0 else 0.0
        
        results = {
            'sigma_ideal': sigma_ideal,
            'sigma_shift': sigma_shift,
            'sigma_exp': sigma_exp,
            'avg_intensity': avg_intensity,
            'baseline': baseline,
            'baseline_removed': baseline_removed,
            'filtered_data': filtered_data,
            'filtered_baseline_removed': filtered_baseline_removed,
            'outlier_mask': outlier_mask,
            'original_count': len(intensities),
            'filtered_count': len(filtered_data),
            'removal_ratio': (len(intensities) - len(filtered_data)) / len(intensities) if len(intensities) > 0 else 0.0
        }
        
        return results
    
    def fine_analysis_export(self, intensities: np.ndarray, radial_bin_index: int, 
                           export_path: Path, xps_value: Optional[float] = None):
        """
        Create CSV file with before/after baseline removal for non-outlier data only
        """
        # Perform fine analysis
        results = self.fine_analysis(intensities)
        
        # Get non-outlier indices
        non_outlier_indices = np.where(results['outlier_mask'])[0]
        
        if len(non_outlier_indices) == 0:
            print(f"No non-outlier data for radial bin {radial_bin_index}, skipping export")
            return
        
        # Create DataFrame with only non-outlier data
        df_export = pd.DataFrame({
            'data_point_index': non_outlier_indices,
            'original_intensity': results['filtered_data'],
            'baseline_removed_intensity': results['filtered_baseline_removed']
        })
        
        # Add metadata
        df_export['radial_bin_index'] = radial_bin_index
        df_export['sigma_ideal'] = results['sigma_ideal']
        df_export['sigma_exp'] = results['sigma_exp']
        if xps_value is not None:
            df_export['xps_value'] = xps_value
        
        # Save to CSV
        df_export.to_csv(export_path, index=False)
        print(f"Exported fine analysis data to {export_path}")
    
    def analyze_single_xps(self, df: pd.DataFrame, xps_value: Optional[float] = None, 
                        xps_group_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Analyze single XPS group and return summary statistics for all radial bins
        Also exports detailed data for radial_bin_080 and saves complete filtered parquet file
        """
        results = []
        
        # Create results directory for this XPS group if path provided
        if xps_group_dir is not None:
            xps_export_dir = xps_group_dir / "radial_analysis"
            xps_export_dir.mkdir(exist_ok=True)
        else:
            xps_export_dir = self.export_base_dir / f"xps_{xps_value:.5f}" if xps_value else self.export_base_dir / "unknown_xps"
            xps_export_dir.mkdir(exist_ok=True)
        
        # Store outlier information for each radial bin and each image
        n_images = len(df)
        n_bins = 512
        outlier_matrix = np.zeros((n_images, n_bins), dtype=bool)  # True = outlier, False = good
        
        print(f"Processing {n_images} images across {n_bins} radial bins...")
        
        # Process all radial bins (000 to 511) and collect outlier information
        for bin_idx in range(512):
            col_name = f'radial_bin_{bin_idx:03d}'
            if col_name in df.columns:
                intensities = df[col_name].values
                
                if len(intensities) > 0 and not np.all(np.isnan(intensities)):
                    # Perform fine analysis
                    analysis_results = self.fine_analysis(intensities)
                    
                    # Store outlier information for this radial bin
                    outlier_matrix[:, bin_idx] = ~analysis_results['outlier_mask']
                    
                    # Store results for summary
                    bin_result = {
                        'radial_bin_index': bin_idx,
                        'avg_intensity': analysis_results['avg_intensity'],
                        'sigma_ideal': analysis_results['sigma_ideal'],
                        'sigma_exp': analysis_results['sigma_exp'],
                        'sigma_shift': analysis_results['sigma_shift'],
                        'original_count': analysis_results['original_count'],
                        'filtered_count': analysis_results['filtered_count'],
                        'removal_ratio': analysis_results['removal_ratio']
                    }
                    results.append(bin_result)
                    
                    # Export detailed data for radial_bin_080
                    if bin_idx == 80:
                        export_filename = f"radial_bin_080_analysis_xps_{xps_value:.5f}.csv" if xps_value else "radial_bin_080_analysis.csv"
                        export_path = xps_export_dir / export_filename
                        self.fine_analysis_export(intensities, bin_idx, export_path, xps_value)
                else:
                    # No valid data for this bin, mark all as outliers
                    outlier_matrix[:, bin_idx] = True
                    
                    # Empty result for this bin
                    bin_result = {
                        'radial_bin_index': bin_idx,
                        'avg_intensity': np.nan,
                        'sigma_ideal': np.nan,
                        'sigma_exp': np.nan,
                        'sigma_shift': np.nan,
                        'original_count': 0,
                        'filtered_count': 0,
                        'removal_ratio': np.nan
                    }
                    results.append(bin_result)
        
        # Calculate which images pass the 90% threshold
        print("Applying 90% threshold filter to images...")
        n_valid_bins_per_image = np.sum(~outlier_matrix, axis=1)  # Count good bins per image
        percentage_good_bins = n_valid_bins_per_image / n_bins
        image_passes_filter = percentage_good_bins >= 0.95  # 95% threshold
        
        n_passing_images = np.sum(image_passes_filter)
        n_removed_images = n_images - n_passing_images
        
        print(f"Image filtering results: {n_passing_images}/{n_images} images passed ({n_passing_images/n_images*100:.1f}%)")
        print(f"Removed {n_removed_images} images that had <95% valid radial bins")
        
        # Create filtered DataFrame with only passing images
        filtered_df = df[image_passes_filter].copy()
        
        # Save complete filtered parquet file
        if xps_value is not None:
            finefilt_filename = f"finefilt_xps_{xps_value:.5f}.parquet"
        else:
            finefilt_filename = "finefilt_xps_unknown.parquet"
        
        if xps_group_dir is not None:
            finefilt_path = xps_group_dir / finefilt_filename
        else:
            finefilt_path = xps_export_dir / finefilt_filename
        
        filtered_df.to_parquet(finefilt_path)
        print(f"Saved complete filtered data to {finefilt_path}")
        
        # Add filtering statistics to results
        for bin_result in results:
            bin_result['images_passing_filter'] = n_passing_images
            bin_result['images_removed'] = n_removed_images
            bin_result['passing_percentage'] = n_passing_images / n_images if n_images > 0 else 0.0
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Save summary CSV
        summary_filename = f"xps_summary_{xps_value:.5f}.csv" if xps_value else "xps_summary_unknown.csv"
        summary_path = xps_export_dir / summary_filename
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved XPS group summary to {summary_path}")
        
        return summary_df
    
    def process_all_xps_groups(self, analysis_directory: str):
        """
        Process all XPS groups in the analysis directory
        """
        analysis_dir = Path(analysis_directory)
        
        # Find all filtered parquet files
        parquet_files = list(analysis_dir.glob("**/filtered_xps_*.parquet"))
        
        if not parquet_files:
            print(f"No filtered parquet files found in {analysis_directory}")
            # Try looking for original files if no filtered ones found
            parquet_files = list(analysis_dir.glob("**/xps_*.parquet"))
            print(f"Found {len(parquet_files)} original parquet files to process")
        
        all_summaries = {}
        
        for file_path in parquet_files:
            try:
                # Extract XPS value from filename
                filename = file_path.name
                if filename.startswith('filtered_xps_'):
                    xps_str = filename[13:-8]  # Remove 'filtered_xps_' and '.parquet'
                else:
                    xps_str = filename[4:-8]  # Remove 'xps_' and '.parquet'
                
                try:
                    xps_value = float(xps_str)
                except ValueError:
                    print(f"Could not extract XPS value from {filename}, using None")
                    xps_value = None
                
                print(f"Processing XPS group: {filename}")
                
                # Load data
                df = pd.read_parquet(file_path)
                
                # Analyze this XPS group
                summary_df = self.analyze_single_xps(df, xps_value, file_path.parent)
                
                # Store summary
                key = f"xps_{xps_value:.5f}" if xps_value else f"file_{file_path.name}"
                all_summaries[key] = summary_df
                
                print(f"Completed processing {filename}: {len(df)} images, {len(summary_df)} radial bins")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all summaries into one master file
        if all_summaries:
            self._create_master_summary(all_summaries, analysis_dir)
        
        return all_summaries
    
    def _create_master_summary(self, all_summaries: Dict, analysis_dir: Path):
        """Create a master summary file combining all XPS groups"""
        master_data = []
        
        for xps_key, summary_df in all_summaries.items():
            # Extract XPS value from key
            if xps_key.startswith('xps_'):
                try:
                    xps_value = float(xps_key[4:])
                except ValueError:
                    xps_value = np.nan
            else:
                xps_value = np.nan
            
            for _, row in summary_df.iterrows():
                master_row = {
                    'xps_value': xps_value,
                    'radial_bin_index': row['radial_bin_index'],
                    'avg_intensity': row['avg_intensity'],
                    'sigma_ideal': row['sigma_ideal'],
                    'sigma_exp': row['sigma_exp'],
                    'sigma_shift': row['sigma_shift'],
                    'filtered_count': row['filtered_count'],
                    'removal_ratio': row['removal_ratio']
                }
                master_data.append(master_row)
        
        master_df = pd.DataFrame(master_data)
        master_path = self.export_base_dir / "master_radial_analysis_summary.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Created master summary file: {master_path}")