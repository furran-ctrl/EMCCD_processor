import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class ResultsFilter:
    def __init__(self, analyze_dir: str):
        """
        Initialize Results Filter.
        
        Args:
            analyze_dir: Path to analysis directory (e.g., 'results/analyze_001')
        """
        self.analyze_dir = Path(analyze_dir)
        self.logger = logging.getLogger(__name__)
        
    def filter_xps_file(self, xps_file: Path) -> None:
        """
        Filter a single XPS parquet file and save filtered version.
        
        Args:
            xps_file: Path to the XPS parquet file
        """
        # Read the data
        df = pd.read_parquet(xps_file)
        
        if df.empty:
            self.logger.warning(f"Empty file: {xps_file}")
            return
        
        # Calculate statistics
        center_x_mean = df['center_x'].mean()
        center_x_std = df['center_x'].std()
        center_y_mean = df['center_y'].mean() 
        center_y_std = df['center_y'].std()
        radial_bin_0_mean = df['radial_bin_000'].mean()
        radial_bin_0_std = df['radial_bin_000'].std()
        
        # Apply 3-sigma filtering
        mask = (
            (df['center_x'] >= center_x_mean - 3 * center_x_std) &
            (df['center_x'] <= center_x_mean + 3 * center_x_std) &
            (df['center_y'] >= center_y_mean - 3 * center_y_std) & 
            (df['center_y'] <= center_y_mean + 3 * center_y_std) &
            (df['radial_bin_000'] >= radial_bin_0_mean - 3 * radial_bin_0_std) &
            (df['radial_bin_000'] <= radial_bin_0_mean + 3 * radial_bin_0_std)
        )
        
        filtered_df = df[mask]
        removed_count = len(df) - len(filtered_df)
        
        # Save filtered file
        output_file = xps_file.parent / f"filtered_{xps_file.name}"
        filtered_df.to_parquet(output_file, index=False)
        
        self.logger.info(f"Filtered {xps_file.name}: {len(filtered_df)}/{len(df)} kept, {removed_count} removed")
        
        return {
            'original_count': len(df),
            'filtered_count': len(filtered_df), 
            'removed_count': removed_count,
            'center_x_stats': (center_x_mean, center_x_std),
            'center_y_stats': (center_y_mean, center_y_std),
            'radial_bin_0_stats': (radial_bin_0_mean, radial_bin_0_std)
        }
    
    def plot_distributions(self, xps_file: Path) -> None:
        """
        Plot distributions of center_x, center_y and radial_bin_000 with 3-sigma lines.
        
        Args:
            xps_file: Path to the XPS parquet file
        """
        df = pd.read_parquet(xps_file)
        
        if df.empty:
            self.logger.warning(f"Cannot plot empty file: {xps_file}")
            return
        
        # Calculate statistics
        center_x_mean, center_x_std = df['center_x'].mean(), df['center_x'].std()
        center_y_mean, center_y_std = df['center_y'].mean(), df['center_y'].std()
        radial_bin_0_mean, radial_bin_0_std = df['radial_bin_000'].mean(), df['radial_bin_000'].std()
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot center_x distribution
        axes[0].hist(df['center_x'], bins=50, alpha=0.7, color='blue')
        axes[0].axvline(center_x_mean, color='red', linestyle='--', label=f'Mean: {center_x_mean:.2f}')
        axes[0].axvline(center_x_mean - 3*center_x_std, color='orange', linestyle=':', label='±3σ')
        axes[0].axvline(center_x_mean + 3*center_x_std, color='orange', linestyle=':')
        axes[0].set_xlabel('Center X')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Center X Distribution')
        axes[0].legend()
        
        # Plot center_y distribution  
        axes[1].hist(df['center_y'], bins=50, alpha=0.7, color='green')
        axes[1].axvline(center_y_mean, color='red', linestyle='--', label=f'Mean: {center_y_mean:.2f}')
        axes[1].axvline(center_y_mean - 3*center_y_std, color='orange', linestyle=':', label='±3σ')
        axes[1].axvline(center_y_mean + 3*center_y_std, color='orange', linestyle=':')
        axes[1].set_xlabel('Center Y')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Center Y Distribution')
        axes[1].legend()
        
        # Plot radial_bin_000 distribution
        axes[2].hist(df['radial_bin_000'], bins=50, alpha=0.7, color='purple')
        axes[2].axvline(radial_bin_0_mean, color='red', linestyle='--', label=f'Mean: {radial_bin_0_mean:.2f}')
        axes[2].axvline(radial_bin_0_mean - 3*radial_bin_0_std, color='orange', linestyle=':', label='±3σ')
        axes[2].axvline(radial_bin_0_mean + 3*radial_bin_0_std, color='orange', linestyle=':')
        axes[2].set_xlabel('Radial Bin 0 Intensity')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Radial Bin 0 Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = xps_file.parent / f"distribution_plot_{xps_file.stem}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved distribution plot: {plot_file}")
    
    def process_all_xps_files(self) -> Dict:
        """
        Process all XPS files matching naming pattern in the analysis directory.
        
        Returns:
            Dictionary with filtering summary statistics
        """
        # Find all XPS directories that match the pattern
        xps_dirs = [d for d in self.analyze_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('xps_')]

        xps_files = []
        for xps_dir in xps_dirs:
            # Look for the parquet file with same name as directory
            expected_file = xps_dir / f"{xps_dir.name}.parquet"
            if expected_file.exists():
                xps_files.append(expected_file)

        # Filter out already processed files
        xps_files = [f for f in xps_files if not f.name.startswith('filtered_')]

        if not xps_files:
            self.logger.warning(f"No XPS files found in {self.analyze_dir}")
            return {}

        self.logger.info(f"Found {len(xps_files)} XPS files to filter")
        
        summary = {}
        
        for xps_file in xps_files:
            self.logger.info(f"Processing {xps_file.name}")
            
            # Create distribution plot
            self.plot_distributions(xps_file)
            
            # Apply filtering
            filter_stats = self.filter_xps_file(xps_file)
            summary[xps_file.stem] = filter_stats
        
        # Generate overall summary
        total_original = sum(stats['original_count'] for stats in summary.values())
        total_filtered = sum(stats['filtered_count'] for stats in summary.values())
        total_removed = total_original - total_filtered
        
        self.logger.info(f"Overall filtering: {total_filtered}/{total_original} kept ({total_removed} removed)")
        
        return {
            'file_summary': summary,
            'total_original': total_original,
            'total_filtered': total_filtered, 
            'total_removed': total_removed,
            'retention_rate': total_filtered / total_original if total_original > 0 else 0
        }