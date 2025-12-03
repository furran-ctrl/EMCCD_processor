import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_intensity_profiles(analysis_dir: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Load all intensity_profile CSV files and organize data.
    
    Parameters:
    -----------
    analysis_dir : str
        Path to the analysis directory
        
    Returns:
    --------
    List of tuples: [(xps_value, avg_intensities, radial_distances), ...]
        - xps_value: XPS value as float
        - avg_intensities: 1D numpy array of average intensity values (512 bins)
        - radial_distances: 1D numpy array of radial distance values (512 bins)
    
    Notes:
    ------
    - For multiple intensity_profile files with same XPS value, selects the latest one
      based on timestamp or highest I value in filename
    - Radial distances are extracted from bin numbers (assuming 1 pixel per bin)
    """
    analysis_path = Path(analysis_dir)
    
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    
    # Find all intensity_profile CSV files recursively
    profile_files = list(analysis_path.glob("**/intensity_profile_xps*.csv"))
    
    if not profile_files:
        logger.warning(f"No intensity_profile files found in {analysis_dir}")
        return []
    
    logger.info(f"Found {len(profile_files)} intensity_profile files")
    
    # Dictionary to store files grouped by XPS value
    xps_groups = {}
    
    # Regex pattern to extract XPS value and I value from filename
    pattern = r"intensity_profile_xps([\d\.]+)_I([\d\.]+)\.csv"
    
    for file_path in profile_files:
        match = re.search(pattern, file_path.name)
        if not match:
            logger.warning(f"Filename doesn't match expected pattern: {file_path.name}")
            continue
        
        xps_value = float(match.group(1))
        i_value = float(match.group(2))
        
        # Store file with its I value
        if xps_value not in xps_groups:
            xps_groups[xps_value] = []
        
        xps_groups[xps_value].append({
            'path': file_path,
            'i_value': i_value,
            'timestamp': file_path.stat().st_mtime  # File modification time
        })
    
    logger.info(f"Grouped into {len(xps_groups)} unique XPS values")
    
    # Process each XPS group
    result_data = []
    
    for xps_value, file_list in sorted(xps_groups.items()):
        # Select the latest file (highest I value, or most recent timestamp)
        if len(file_list) > 1:
            # Sort by I value (descending), then by timestamp (descending)
            file_list.sort(key=lambda x: (x['i_value'], x['timestamp']), reverse=True)
            logger.info(f"XPS {xps_value:.5f}: {len(file_list)} files found, "
                       f"selecting I={file_list[0]['i_value']:.5f}")
        
        selected_file = file_list[0]['path']
        
        try:
            # Load the CSV file
            df = pd.read_csv(selected_file)
            
            # Validate required columns
            required_columns = ['bin_number', 'average']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in {selected_file.name}")
                continue
            
            # Sort by bin_number to ensure correct order
            df = df.sort_values('bin_number')
            
            # Extract average intensities
            avg_intensities = df['average'].values.astype(np.float64)
            
            # Create radial distances (assuming 1 pixel per bin starting from 0)
            # You might want to adjust this based on your actual pixel-to-distance calibration
            radial_distances = df['bin_number'].values.astype(np.float64)
            
            # Alternative: If you have actual distance calibration, you could do:
            # radial_distances = df['bin_number'].values * calibration_factor
            
            # Validate data shape (should be 512 bins)
            if len(avg_intensities) != 512:
                logger.warning(f"Expected 512 bins, got {len(avg_intensities)} in {selected_file.name}")
            
            # Add to results
            result_data.append((xps_value, avg_intensities, radial_distances))
            
            logger.debug(f"Loaded XPS {xps_value:.5f}: {len(avg_intensities)} bins, "
                        f"I={file_list[0]['i_value']:.5f}")
            
        except Exception as e:
            logger.error(f"Error loading {selected_file}: {e}")
            continue
    
    # Sort results by XPS value
    result_data.sort(key=lambda x: x[0])
    
    logger.info(f"Successfully loaded {len(result_data)} intensity profiles")
    
    return result_data


def load_intensity_profiles_with_metadata(analysis_dir: str) -> List[dict]:
    """
    Alternative version that returns more metadata.
    
    Returns:
    --------
    List of dictionaries with complete information:
    [
        {
            'xps_value': float,
            'avg_intensities': np.ndarray,
            'std_intensities': np.ndarray,  # if available
            'radial_distances': np.ndarray,
            'i_value': float,
            'filename': str,
            'num_bins': int
        },
        ...
    ]
    """
    analysis_path = Path(analysis_dir)
    pattern = r"intensity_profile_xps([\d\.]+)_I([\d\.]+)\.csv"
    
    # Find and group files
    profile_files = list(analysis_path.glob("**/intensity_profile_xps*.csv"))
    xps_groups = {}
    
    for file_path in profile_files:
        match = re.search(pattern, file_path.name)
        if not match:
            continue
        
        xps_value = float(match.group(1))
        i_value = float(match.group(2))
        
        if xps_value not in xps_groups:
            xps_groups[xps_value] = []
        
        xps_groups[xps_value].append({
            'path': file_path,
            'i_value': i_value,
            'timestamp': file_path.stat().st_mtime
        })
    
    # Process each group
    result_list = []
    
    for xps_value, file_list in sorted(xps_groups.items()):
        # Select latest file
        file_list.sort(key=lambda x: (x['i_value'], x['timestamp']), reverse=True)
        selected = file_list[0]
        
        try:
            df = pd.read_csv(selected['path'])
            df = df.sort_values('bin_number')
            
            # Create result dictionary
            result = {
                'xps_value': xps_value,
                'avg_intensities': df['average'].values.astype(np.float64),
                'radial_distances': df['bin_number'].values.astype(np.float64),
                'i_value': selected['i_value'],
                'filename': selected['path'].name,
                'filepath': str(selected['path']),
                'num_bins': len(df)
            }
            
            # Add std if available
            if 'std' in df.columns:
                result['std_intensities'] = df['std'].values.astype(np.float64)
            
            result_list.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {selected['path'].name}: {e}")
            continue
    
    return result_list


# Example usage and helper functions
def save_combined_profiles(data_list: List[Tuple[float, np.ndarray, np.ndarray]], 
                          output_path: str):
    """
    Save combined intensity profiles to a CSV file.
    
    Parameters:
    -----------
    data_list : List from load_intensity_profiles()
    output_path : str
        Path to save the combined CSV
    """
    all_data = []
    
    for xps_value, avg_intensities, radial_distances in data_list:
        for bin_num, avg_intensity, distance in zip(range(len(avg_intensities)), 
                                                   avg_intensities, 
                                                   radial_distances):
            all_data.append({
                'xps_value': xps_value,
                'bin_number': bin_num,
                'radial_distance': distance,
                'average_intensity': avg_intensity
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved combined profiles to {output_path}")


def plot_intensity_profiles(data_list: List[Tuple[float, np.ndarray, np.ndarray]], 
                           max_profiles: int = 10):
    """
    Quick plot of intensity profiles (requires matplotlib).
    
    Parameters:
    -----------
    data_list : List from load_intensity_profiles()
    max_profiles : int
        Maximum number of profiles to plot
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for i, (xps_value, avg_intensities, radial_distances) in enumerate(data_list):
            if i >= max_profiles:
                break
            
            plt.plot(radial_distances, avg_intensities, 
                    label=f'XPS={xps_value:.2f}', alpha=0.7)
        
        plt.xlabel('Radial Distance (pixels)')
        plt.ylabel('Average Intensity')
        plt.title('Intensity Profiles by XPS Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping plot.")


# Main execution example
if __name__ == "__main__":
    # Example usage
    analysis_directory = r"C:\Users\86177\Desktop\streaking\analysis_parallel_time"
    
    # Load all intensity profiles
    profiles = load_intensity_profiles(analysis_directory)
    
    # Display summary
    # print(f"Loaded {len(profiles)} intensity profiles:")
    # for xps_value, intensities, distances in profiles:
    #     print(f"  XPS={xps_value:.5f}: {len(intensities)} bins, "
    #           f"intensity range=[{intensities.min():.2e}, {intensities.max():.2e}]")
    
    # Save combined data
    # if profiles:
    #     save_combined_profiles(profiles, "combined_intensity_profiles.csv")
        
    #     # Quick plot (if matplotlib available)
    #     plot_intensity_profiles(profiles)