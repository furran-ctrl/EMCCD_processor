import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_similar_xps_files(source_dirs: List[str], 
                           output_dir: str, 
                           tolerance: float = 0.001,
                           merge_method: str = 'mean') -> Dict[str, List[Tuple[float, float, int]]]:
    """
    Merge parquet files with similar XPS values across multiple analysis directories.
    
    Parameters:
    -----------
    source_dirs : List[str]
        List of source analysis directories
    output_dir : str
        Output directory for merged files
    tolerance : float
        Maximum difference in XPS values to consider as similar (default: 0.001)
    merge_method : str
        How to handle multiple files: 'mean', 'median', or 'closest'
    
    Returns:
    --------
    Dictionary mapping merged XPS values to list of (original_xps, weight, n_rows)
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect all parquet files
    all_files = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue
        
        # Find all xps_*.parquet files
        parquet_files = list(source_path.glob("**/normalized_xps_*.parquet"))
        
        for file_path in parquet_files:
            try:
                # Extract XPS value from filename
                # Expecting format: xps_{value}.parquet or xps_{value}/xps_{value}.parquet
                if file_path.parent.name.startswith('normalized_xps_'):
                    # Format: xps_{value}/xps_{value}.parquet
                    xps_str = file_path.parent.name.replace('normalized_xps_', '')
                else:
                    # Format: xps_{value}.parquet
                    xps_str = file_path.stem.replace('normalized_xps_', '')
                
                xps_value = float(xps_str)
                
                # Load metadata to get row count
                try:
                    df = pd.read_parquet(file_path)
                    n_rows = len(df)
                    all_files.append({
                        'path': file_path,
                        'xps': xps_value,
                        'n_rows': n_rows,
                        'source_dir': source_dir,
                        'df': df  # Keep DataFrame for merging
                    })
                    logger.debug(f"Found: {file_path}, XPS={xps_value:.5f}, rows={n_rows}")
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    
            except ValueError as e:
                logger.warning(f"Could not parse XPS value from {file_path}: {e}")
    
    if not all_files:
        logger.error("No valid parquet files found")
        return {}
    
    logger.info(f"Found {len(all_files)} parquet files")
    
    # Step 2: Group files by similar XPS values
    # Sort by XPS value
    all_files.sort(key=lambda x: x['xps'])
    
    groups = []
    current_group = []
    current_ref_xps = None
    
    for file_info in all_files:
        if current_ref_xps is None:
            # Start first group
            current_group = [file_info]
            current_ref_xps = file_info['xps']
        elif abs(file_info['xps'] - current_ref_xps) <= tolerance:
            # Add to current group
            current_group.append(file_info)
        else:
            # Save current group and start new one
            groups.append(current_group)
            current_group = [file_info]
            current_ref_xps = file_info['xps']
    
    # Don't forget the last group
    if current_group:
        groups.append(current_group)
    
    logger.info(f"Grouped into {len(groups)} XPS groups (tolerance={tolerance})")
    
    # Step 3: Process each group
    merge_info = {}
    
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        
        # Calculate merged XPS value
        xps_values = [f['xps'] for f in group]
        row_counts = [f['n_rows'] for f in group]
        
        if merge_method == 'mean':
            # Weighted mean by row count
            total_rows = sum(row_counts)
            weights = [rc / total_rows for rc in row_counts]
            merged_xps = sum(xps * weight for xps, weight in zip(xps_values, weights))
        elif merge_method == 'median':
            merged_xps = np.median(xps_values)
        elif merge_method == 'closest':
            # Use the XPS value from file with most rows
            max_idx = np.argmax(row_counts)
            merged_xps = xps_values[max_idx]
        else:
            raise ValueError(f"Unknown merge_method: {merge_method}")
        
        # Format merged XPS with 5 decimal places
        merged_xps_formatted = f"{merged_xps:.5f}"
        
        # Create output directory for this XPS group
        xps_output_dir = output_path / f"xps_{merged_xps_formatted}"
        xps_output_dir.mkdir(exist_ok=True)
        
        # Merge DataFrames
        dfs_to_merge = [f['df'] for f in group]
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # Update xps_value column to merged value
        if 'xps_value' in merged_df.columns:
            merged_df['xps_value'] = merged_xps
        
        # Save merged parquet file
        output_file = xps_output_dir / f"xps_{merged_xps_formatted}.parquet"
        merged_df.to_parquet(output_file, index=False)
        
        # Store merge information
        merge_info[merged_xps_formatted] = [
            (f['xps'], f['n_rows'], str(f['path'])) for f in group
        ]
        
        logger.info(f"Group {i+1}: Merged {len(group)} files → XPS={merged_xps_formatted}, "
                   f"rows={len(merged_df)}")
        
        # Log details
        logger.debug(f"  Original XPS values: {[f'{x:.5f}' for x in xps_values]}")
        logger.debug(f"  Row counts: {row_counts}")
    
    # Step 4: Save merge log
    log_file = output_path / "merge_log.csv"
    
    log_data = []
    for merged_xps, files in merge_info.items():
        for orig_xps, n_rows, filepath in files:
            log_data.append({
                'merged_xps': float(merged_xps),
                'original_xps': orig_xps,
                'difference': abs(float(merged_xps) - orig_xps),
                'rows': n_rows,
                'source_file': filepath
            })
    
    if log_data:
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_file, index=False, float_format='%.5f')
        logger.info(f"Saved merge log to {log_file}")
    
    # Step 5: Copy any other files (optional)
    logger.info("Copying other files...")
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        for item in source_path.iterdir():
            if item.is_file() and not item.name.startswith('xps_'):
                shutil.copy2(item, output_path / item.name)
                logger.debug(f"Copied: {item.name}")
    
    logger.info(f"Merge complete! Output directory: {output_dir}")
    logger.info(f"Merged {len(all_files)} files into {len(groups)} XPS groups")
    
    return merge_info