#This is an alternative version of diffraction_normalizer with bootstrap for filtering.
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import csv
import statistics
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress bootstrap warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiffractionNormalizer:
    """
    Second-pass normalizer for diffraction data.
    Processes filtered_xps_{}.parquet files from the first filtering step.
    """
    
    # Radial bins to use for normalization factor calculation
    NORMALIZATION_BINS = [f"radial_bin_{i:03d}" for i in range(60, 201, 20)]  # 60, 80, ..., 200
    AVG_TUNING_BINS = [f"radial_bin_{i:03d}" for i in range(330, 411, 20)]  # q8~10 at 0.024 A^-1/pixel
    
    def __init__(self, analysis_dir: str):
        """
        Initialize the normalizer with the analysis directory.
        
        Parameters:
        -----------
        analysis_dir : str
            Path to the analysis directory (e.g., 'results/analysis_{TIMESTAMP}')
        """
        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
        
        logger.info(f"Initialized DiffractionNormalizer for directory: {analysis_dir}")
    
    def find_filtered_parquet_files(self) -> List[Path]:
        """
        Find all filtered parquet files in the analysis directory.
        
        Returns:
        --------
        List of Path objects for each filtered_xps_{VALUE}.parquet file
        """
        filtered_files = list(self.analysis_dir.glob("**/filtered_xps_*.parquet"))
        
        logger.info(f"Found {len(filtered_files)} filtered parquet files")
        return sorted(filtered_files)
    
    def load_filtered_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load filtered parquet file.
        
        Parameters:
        -----------
        file_path : Path
            Path to the filtered parquet file
            
        Returns:
        --------
        pandas DataFrame or None if file cannot be loaded
        """
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            
            # Verify required columns exist
            required_columns = ['filename'] + self.NORMALIZATION_BINS
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in {file_path.name}: {missing_columns}")
                return None
            
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def calculate_normalization_factors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Calculate normalization factor for each row.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with radial bin columns
            
        Returns:
        --------
        (df_with_factors, avg_norm_factor)
        df_with_factors: DataFrame with added 'norm_factor' column
        avg_norm_factor: Average normalization factor
        """
        # Sum the specified radial bins for each row
        norm_factors = df[self.NORMALIZATION_BINS].sum(axis=1)
        
        # Add normalization factor column
        df_with_factors = df.copy()
        df_with_factors['norm_factor'] = norm_factors
        
        # Calculate average normalization factor
        avg_norm_factor = statistics.harmonic_mean(norm_factors)
        
        logger.info(f"Calculated normalization factors: avg={avg_norm_factor:.5f}, "
                   f"min={norm_factors.min():.5f}, max={norm_factors.max():.5f}")
        
        return df_with_factors, avg_norm_factor
    
    def calculate_norm_factor_bootstrap_ci(self, norm_factors: pd.Series, n_bootstrap: int = 10000) -> Dict[str, float]:
        """
        Calculate 95% confidence interval for normalization factors using bootstrap.
        
        Parameters:
        -----------
        norm_factors : pd.Series
            Series of normalization factors
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns:
        --------
        Dictionary with bootstrap statistics and CI
        """
        # Convert to numpy array for bootstrap
        data = (norm_factors.values,)
        
        try:
            # Calculate bootstrap confidence interval
            bootstrap_result = bootstrap(
                data, 
                np.mean, 
                n_resamples=n_bootstrap,
                confidence_level=0.95,
                method='BCa'  # Bias-corrected and accelerated
            )
            
            ci_lower = bootstrap_result.confidence_interval.low
            ci_upper = bootstrap_result.confidence_interval.high
            
            # Calculate additional statistics
            median_val = np.median(norm_factors)
            mean_val = np.mean(norm_factors)
            std_val = np.std(norm_factors)
            
            thresholds = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': len(norm_factors),
                'n_bootstrap': n_bootstrap
            }
            
            logger.info(f"Bootstrap 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}], "
                    f"mean={mean_val:.2f}, std={std_val:.2f}, n={len(norm_factors)}")
            
            return thresholds
            
        except Exception as e:
            logger.error(f"Bootstrap calculation failed: {e}")
            # Fall back to MAD method if bootstrap fails
            return self.calculate_norm_factor_thresholds_fallback(norm_factors)

    def calculate_norm_factor_thresholds_fallback(self, norm_factors: pd.Series) -> Dict[str, float]:
        """
        Fallback method using MAD if bootstrap fails.
        """
        # Calculate median
        median_val = np.median(norm_factors)
        
        # Calculate MAD
        deviations = np.abs(norm_factors - median_val)
        mad = np.median(deviations)
        
        # Convert MAD to sigma (1.4826 for normal distribution)
        sigma = mad * 1.4826
        
        # Calculate ±2σ thresholds as approximate 95% CI
        lower_threshold = median_val - 2 * sigma
        upper_threshold = median_val + 2 * sigma
        
        thresholds = {
            'mean': np.mean(norm_factors),
            'median': median_val,
            'std': np.std(norm_factors),
            'ci_lower': lower_threshold,
            'ci_upper': upper_threshold,
            'method': 'MAD_fallback',
            'n_samples': len(norm_factors)
        }
        
        logger.warning(f"Using MAD fallback: 95% CI approx=[{lower_threshold:.2f}, {upper_threshold:.2f}]")
        
        return thresholds

    def filter_by_norm_factor_bootstrap(self, df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
        """
        Filter rows based on bootstrap 95% confidence interval.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'norm_factor' column
        thresholds : dict
            Dictionary with threshold values from bootstrap
            
        Returns:
        --------
        Filtered DataFrame
        """
        # Create mask for rows within 95% CI
        mask = (df['norm_factor'] >= thresholds['ci_lower']) & \
            (df['norm_factor'] <= thresholds['ci_upper'])
        
        removed_count = (~mask).sum()
        retained_count = mask.sum()
        
        if removed_count > 0:
            logger.info(f"Bootstrap CI filtering: removed {removed_count} rows ({removed_count/len(df)*100:.1f}%), "
                    f"retained {retained_count} rows")
            
            # Log details about removed values
            removed_factors = df.loc[~mask, 'norm_factor']
            logger.debug(f"Removed norm factors: min={removed_factors.min():.2f}, "
                        f"max={removed_factors.max():.2f}, "
                        f"mean={removed_factors.mean():.2f}")
            
            # Log which side of CI the outliers are on
            below_ci = (df['norm_factor'] < thresholds['ci_lower']).sum()
            above_ci = (df['norm_factor'] > thresholds['ci_upper']).sum()
            logger.debug(f"Outliers: {below_ci} below CI, {above_ci} above CI")
        else:
            logger.info("All rows passed bootstrap CI filtering")
        
        return df[mask].copy()

    def normalize_radial_profiles(self, df: pd.DataFrame, avg_norm_factor: float) -> pd.DataFrame:
        """
        Normalize radial profiles by scaling each row.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'norm_factor' column and radial_bin_* columns
        avg_norm_factor : float
            Average normalization factor to scale to
            
        Returns:
        --------
        DataFrame with normalized radial bins
        """
        # Create a copy to avoid modifying the original
        df_normalized = df.copy()
        
        # Identify radial bin columns (excluding metadata columns)
        radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
        
        # Calculate scaling factor for each row
        scaling_factors = avg_norm_factor / df['norm_factor']
        #scaling_factors = 100 / df['norm_factor']

        # Apply scaling to all radial bins
        for col in radial_columns:
            df_normalized[col] = df[col] * scaling_factors
        
        logger.info(f"Normalized radial profiles using scaling factors: "
                   f"avg_scale={scaling_factors.mean():.5f}")
        
        return df_normalized
    
    def save_normalization_factors(self, df: pd.DataFrame, xps_dir: Path, xps_value: float):
        """
        Save normalization factors to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'filename' and 'norm_factor' columns
        xps_dir : Path
            Directory to save the CSV file in
        xps_value : float
            XPS value for this group
        """
        # Create CSV filename
        csv_filename = f"norm_factors_xps{xps_value:.5f}.csv"
        csv_path = xps_dir / csv_filename
        
        # Select only filename and norm_factor columns
        factor_df = df[['filename', 'norm_factor']].copy()
        
        # Sort by filename for consistency
        factor_df = factor_df.sort_values('filename')
        
        # Save to CSV
        factor_df.to_csv(csv_path, index=False, float_format='%.5f')
        
        logger.info(f"Saved normalization factors to {csv_filename} "
                   f"({len(factor_df)} entries)")
    
    def calculate_intensity_profile_stats(self, df_normalized: pd.DataFrame, 
                                        avg_norm_factor: float,
                                        xps_dir: Path, xps_value: float,
                                        n_bootstrap: int = 5000):
        """
        Calculate bootstrap statistics for each radial bin and save to CSV.
        
        Parameters:
        -----------
        df_normalized : pd.DataFrame
            Normalized DataFrame with radial_bin_* columns
        avg_norm_factor : float
            Average normalization factor
        xps_dir : Path
            Directory to save the CSV file in
        xps_value : float
            XPS value for this group
        n_bootstrap : int
            Number of bootstrap samples
        """
        # Identify radial bin columns
        radial_columns = [col for col in df_normalized.columns if col.startswith('radial_bin_')]
        
        # Sort columns to ensure consistent bin order
        radial_columns.sort()
        
        # Initialize lists for results
        averages = []
        stds = []
        sems = []  # Standard Error of the Mean
        ci_lowers = []
        ci_uppers = []
        bin_numbers = []
        
        logger.info(f"Calculating bootstrap statistics for {len(radial_columns)} bins "
                f"(n_bootstrap={n_bootstrap})...")
        
        for i, col in enumerate(radial_columns):
            # Extract bin number from column name
            bin_num = int(col.replace('radial_bin_', ''))
            
            # Get data for this bin
            data = df_normalized[col].values
            
            # Calculate basic statistics
            avg_val = np.mean(data)
            std_val = np.std(data)
            sem_val = std_val / np.sqrt(len(data))
            
            # Calculate bootstrap 95% CI for the mean
            try:
                bootstrap_result = bootstrap(
                    (data,), 
                    np.mean, 
                    n_resamples=n_bootstrap,
                    confidence_level=0.9995,
                    method='BCa'
                )
                ci_lower = bootstrap_result.confidence_interval.low
                ci_upper = bootstrap_result.confidence_interval.high
            except Exception as e:
                logger.warning(f"Bootstrap failed for bin {bin_num}: {e}")
                # Use parametric approximation if bootstrap fails
                ci_lower = avg_val - 1.96 * sem_val
                ci_upper = avg_val + 1.96 * sem_val
            
            # Store results
            averages.append(avg_val)
            stds.append(std_val)
            sems.append(sem_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            bin_numbers.append(bin_num)
            
            # Log progress for every 50 bins
            # if (i + 1) % 50 == 0:
            #     logger.debug(f"Processed {i + 1}/{len(radial_columns)} bins")
        
        # Create DataFrame for statistics
        stats_df = pd.DataFrame({
            'bin_number': bin_numbers,
            'average': averages,
            'std': stds,
            'sem': sems,
            'ci_lower': ci_lowers,
            'ci_upper': ci_uppers
        })
        
        # Create filename with intensity profile
        profile_filename = f"intensity_profile_xps{xps_value:.5f}_I{avg_norm_factor:.5f}.csv"
        profile_path = xps_dir / profile_filename
        
        # Save to CSV
        stats_df.to_csv(profile_path, index=False, float_format='%.5f')
        
        logger.info(f"Saved bootstrap intensity profile to {profile_filename} "
                f"({len(stats_df)} bins)")
        
        return stats_df
    
    def process_xps_group(self, filtered_file: Path, n_bootstrap: int = 10000) -> Tuple[bool, str]:
        """
        Process a single XPS group with bootstrap filtering and statistics.
        """
        try:
            # Extract XPS value from filename
            filename = filtered_file.stem  # 'filtered_xps_188.94000'
            xps_value_str = filename.replace('filtered_xps_', '')
            xps_value = float(xps_value_str)
            
            logger.info(f"Processing XPS group {xps_value:.5f} from {filtered_file.name}")
            
            # Load filtered data
            df = self.load_filtered_data(filtered_file)
            if df is None:
                return False, f"Failed to load data from {filtered_file.name}"
            
            initial_count = len(df)
            
            # Step 1: Calculate normalization factors
            df_with_factors, avg_norm_factor = self.calculate_normalization_factors(df)
            
            # Step 2: Filter based on norm factor 95% CI bootstrap
            bootstrap_thresholds = self.calculate_norm_factor_bootstrap_ci(
                df_with_factors['norm_factor'], 
                n_bootstrap=n_bootstrap
            )
            df_filtered = self.filter_by_norm_factor_bootstrap(df_with_factors, bootstrap_thresholds)
            
            # Check if we have enough data after filtering
            if len(df_filtered) < 10:  # Minimum for meaningful bootstrap
                logger.warning(f"Insufficient data after bootstrap filtering: {len(df_filtered)} rows")
                return False, f"Insufficient data after bootstrap filtering ({len(df_filtered)} rows)"
            
            # Update average norm factor based on filtered data
            avg_norm_factor_filtered = df_filtered['norm_factor'].mean()
            
            logger.info(f"Bootstrap filtering: {initial_count} → {len(df_filtered)} rows, "
                    f"retention={len(df_filtered)/initial_count*100:.1f}%, "
                    f"avg_norm_factor: {avg_norm_factor:.2f} → {avg_norm_factor_filtered:.2f}")
            
            # Step 3: Save normalization factors to CSV (using filtered data)
            self.save_normalization_factors(df_filtered, filtered_file.parent, xps_value)
            
            # Step 4: Normalize radial profiles (using filtered data and updated avg)
            df_normalized = self.normalize_radial_profiles(df_filtered, avg_norm_factor_filtered)
            
            # Step 5: Calculate intensity profile statistics with bootstrap and save to CSV
            stats_df = self.calculate_intensity_profile_stats(
                df_normalized, 
                avg_norm_factor_filtered, 
                filtered_file.parent, 
                xps_value,
                n_bootstrap=n_bootstrap
            )
            
            # Save additional bootstrap info
            bootstrap_info = {
                'xps_value': xps_value,
                'initial_samples': initial_count,
                'filtered_samples': len(df_filtered),
                'avg_norm_factor_before': avg_norm_factor,
                'avg_norm_factor_after': avg_norm_factor_filtered,
                'bootstrap_ci_lower': bootstrap_thresholds['ci_lower'],
                'bootstrap_ci_upper': bootstrap_thresholds['ci_upper'],
                'bootstrap_method': 'BCa',
                'n_bootstrap': n_bootstrap
            }
            
            # Save bootstrap info to JSON
            import json
            info_filename = f"bootstrap_info_xps{xps_value:.5f}.json"
            info_path = filtered_file.parent / info_filename
            with open(info_path, 'w') as f:
                json.dump(bootstrap_info, f, indent=2, default=str)
            
            # Optional: Save filtered data with norm factors
            filtered_with_factors_filename = f"filtered_with_factors_xps{xps_value:.5f}.parquet"
            filtered_with_factors_path = filtered_file.parent / filtered_with_factors_filename
            df_filtered.to_parquet(filtered_with_factors_path, index=False)
            logger.info(f"Saved filtered data with norm factors to {filtered_with_factors_filename}")
            
            # Log summary
            logger.info(f"Successfully processed XPS {xps_value:.5f}: "
                    f"{initial_count} → {len(df_filtered)} files, "
                    f"avg_norm_factor={avg_norm_factor_filtered:.5f}")
            
            return True, (f"Processed {len(df_filtered)}/{initial_count} files, "
                        f"CI=[{bootstrap_thresholds['ci_lower']:.2f},{bootstrap_thresholds['ci_upper']:.2f}], "
                        f"avg_norm_factor={avg_norm_factor_filtered:.5f}")
            
        except Exception as e:
            logger.error(f"Error processing {filtered_file.name}: {e}")
            return False, str(e)
    
    def run_normalization(self, n_bootstrap: int = 10000) -> dict:
        """
        Run the normalization process with bootstrap statistics.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples for CI calculation
            
        Returns:
        --------
        Dictionary with processing results for each XPS group
        """
        results = {}
        
        # Find all filtered parquet files
        filtered_files = self.find_filtered_parquet_files()
        
        if not filtered_files:
            logger.error("No filtered parquet files found. Run the first filter step first.")
            return results
        
        logger.info(f"Using bootstrap with n={n_bootstrap} samples")
        
        # Process each XPS group
        for filtered_file in filtered_files:
            logger.info(f"Processing {filtered_file.name}...")
            success, message = self.process_xps_group(filtered_file, n_bootstrap)
            results[filtered_file.name] = {
                'success': success,
                'message': message,
                'file_path': str(filtered_file)
            }
        
        # Print summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"Normalization complete: {successful}/{len(results)} groups successful")
        
        return results
    
    def generate_summary_report(self, results: dict, output_dir: Optional[Path] = None):
        """
        Generate a summary report of all processed XPS groups.
        
        Parameters:
        -----------
        results : dict
            Results from run_normalization()
        output_dir : Optional[Path]
            Directory to save summary report (default: analysis_dir)
        """
        if output_dir is None:
            output_dir = self.analysis_dir
        
        summary_data = []
        
        for filename, result in results.items():
            if result['success']:
                # Extract XPS value from filename
                try:
                    xps_value = float(filename.replace('filtered_xps_', '').replace('.parquet', ''))
                    
                    # Find the intensity profile file
                    xps_dir = Path(result['file_path']).parent
                    profile_files = list(xps_dir.glob(f"intensity_profile_xps{xps_value:.5f}_I*.csv"))
                    
                    if profile_files:
                        # Extract average norm factor from filename
                        profile_filename = profile_files[0].name
                        # Find pattern _I{value}.csv
                        import re
                        match = re.search(r'_I([\d\.]+)\.csv', profile_filename)
                        avg_norm_factor = float(match.group(1)) if match else None
                        
                        # Load stats to get number of bins
                        stats_df = pd.read_csv(profile_files[0])
                        num_bins = len(stats_df)
                        
                        summary_data.append({
                            'xps_value': xps_value,
                            'avg_norm_factor': avg_norm_factor,
                            'num_files': 'N/A',  # Could load from norm_factors CSV
                            'num_bins': num_bins,
                            'status': 'Success'
                        })
                except Exception as e:
                    logger.warning(f"Could not extract details for {filename}: {e}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('xps_value')
            
            # Save summary to CSV
            summary_path = output_dir / "normalization_summary.csv"
            summary_df.to_csv(summary_path, index=False, float_format='%.5f')
            
            logger.info(f"Saved normalization summary to {summary_path}")
            
            # Print summary table
            print("\n" + "="*80)
            print("NORMALIZATION SUMMARY")
            print("="*80)
            print(summary_df.to_string(index=False))
            print()
            
            return summary_df
        else:
            logger.warning("No successful results to summarize")
            return None
