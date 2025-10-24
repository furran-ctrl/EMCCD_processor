import tifffile as tiff
import numpy as np
import os

from typing import List
from pathlib import Path
#automatically handles path separators across different operating systems.

def TiffLoader(directory: str, filename: str) -> np.ndarray:
    '''
    filename must include .tiff suffix!!!
    '''
    # Convert to Path object
    dir_path = Path(directory)
    # Construct full file path
    file_path = dir_path / filename
    # Load and return the TIFF file
    try:
        # Load the TIFF file
        image = tiff.imread(file_path)
        return image
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def TiffLoaderBatch(directory: str) -> List[np.ndarray]:
    '''
    Loads all tiff under a /dir and returns a list of ndarray
    '''
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
            image = tiff.imread(file_path)
            image_list.append(image)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    return image_list