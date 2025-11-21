import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
import tables  # PyTables

@dataclass
class ProcessingConfig:
    """Dataclass to store all processing parameters."""
    result_directory: str
    background_directory: str
    data_mask_directory: str
    xps_grouping_param: List[float]  # [threshold, tolerance]
    xray_removal_param: List[float]  # [chunk_size, sigma_threshold, beam_threshold]
    center_fitting_param: List[float]  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param: List[float]  # [radius, num_bins]
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

# External function to save configuration
def save_processing_config(config: ProcessingConfig, save_path: Path) -> None:
    """
    Save processing configuration to file.
    
    Args:
        config: ProcessingConfig dataclass instance
        save_path: Path to save configuration
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    config_df = pd.DataFrame([asdict(config)])
    config_df.to_csv(save_path, index=False)
    
    # Also save as JSON
    '''json_path = save_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)'''

# External function to load configuration
def load_processing_config(load_path: Path) -> ProcessingConfig:
    """
    Load processing configuration from file.
    
    Args:
        load_path: Path to configuration file
        
    Returns:
        ProcessingConfig dataclass instance
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Config file not found: {load_path}")
    
    config_df = pd.read_csv(load_path)
    config_dict = config_df.iloc[0].to_dict()
    
    return ProcessingConfig.from_dict(config_dict)

class ProcessedResult:
    """Container for processed image results."""
    def __init__(self, 
                 filename: str,
                 center: Tuple[float, float],
                 total_count: float,
                 radial_profile: np.ndarray,
                 xps_value: float):
        self.filename = filename
        self.center = center
        self.total_count = total_count
        self.radial_profile = radial_profile
        self.xps_value = xps_value

class HDF5DataStore:
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def save_results(self, results: List['ProcessedResult']):
        """Save results to HDF5 file."""
        with tables.open_file(self.filepath, 'w') as f:
            # Create table for metadata
            metadata_dtype = np.dtype([
                ('filename', 'S256'),  # string up to 256 chars
                ('center_x', 'f8'),
                ('center_y', 'f8'), 
                ('total_count', 'f8'),
                ('xps_value', 'f8')
            ])
            
            metadata_table = f.create_table(f.root, 'metadata', metadata_dtype)
            
            # Create array for radial profiles
            radial_profiles = f.create_earray(f.root, 'radial_profiles', 
                                            tables.Float64Atom(), 
                                            shape=(0, 512))  # expandable
            
            # Store data
            for result in results:
                # Add to metadata table
                row = metadata_table.row
                row['filename'] = result.filename.encode()
                row['center_x'] = result.center[0]
                row['center_y'] = result.center[1]
                row['total_count'] = result.total_count
                row['xps_value'] = result.xps_value
                row.append()
                
                # Add to radial profiles array
                radial_profiles.append([result.radial_profile])
            
            metadata_table.flush()
    
    def load_results(self) -> pd.DataFrame:
        """Load results as pandas DataFrame."""
        with tables.open_file(self.filepath, 'r') as f:
            # Read metadata
            metadata = f.root.metadata.read()
            radial_profiles = f.root.radial_profiles.read()
            
        # Convert to DataFrame
        df = pd.DataFrame(metadata)
        df['filename'] = df['filename'].str.decode('utf-8')
        
        # Add radial profiles as separate columns
        for i in range(512):
            df[f'radial_bin_{i}'] = radial_profiles[:, i]
        
        return df