import tifffile 
import numpy as np
import os

from typing import List
from pathlib import Path
#automatically handles path separators across different operating systems.

def TiffLoader(directory: str, filename: str) -> np.ndarray:
    """
    Load a TIFF file from the specified directory and return as numpy ndarray.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing the TIFF file
    filename : str
        Name of the TIFF file (with or without extension)
    
    Returns:
    --------
    np.ndarray
        The loaded TIFF image as a numpy array
    """
    # Ensure the filename has .tiff or .tif extension
    if not filename.lower().endswith(('.tiff', '.tif')):
        # Try adding extensions if not present
        for ext in ['.tiff', '.tif']:
            potential_path = os.path.join(directory, filename + ext)
            if os.path.exists(potential_path):
                filename = filename + ext
                break
        else:
            # If no file found with extensions, use the original filename
            pass
    
    # Construct full file path
    dir_path = Path(directory)
    file_path = os.path.join(dir_path, filename)
    
    # Load and return the TIFF file
    return tifffile.imread(file_path)

def TiffLoaderBatch(directory: str) -> List[np.ndarray]:
    """
    Load all TIFF files from the specified directory and return as list of numpy ndarrays.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing TIFF files
    
    Returns:
    --------
    List[np.ndarray]
        List of loaded TIFF images as numpy arrays
    """
    # Convert to Path object for better handling
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    
    # Find all TIFF files in directory
    file_paths = []
    for ext in ['*.tiff', '*.tif']:
        file_paths.extend(dir_path.glob(ext))
    if not file_paths:
        raise FileNotFoundError(f"No TIFF files found in {directory}")
    print(f"Found {len(file_paths)} TIFF files")
    
    # Load all TIFF files
    image_list = []
    for file_path in file_paths:
        try:
            image = tifffile.imread(file_path)
            image_list.append(image)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    return image_list