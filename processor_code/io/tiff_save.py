#Temp!!

import tifffile
import numpy as np
from pathlib import Path
from typing import Union, Optional

def save_as_tiff(array: np.ndarray, 
                 filepath: Union[str, Path],
                 metadata: Optional[dict] = None,
                 compress: bool = True) -> None:
    """
    Save a numpy ndarray as a TIFF file.
    
    Parameters:
    -----------
    array : np.ndarray
        The numpy array to save as TIFF
    filepath : str or Path
        Path where the TIFF file will be saved
    metadata : dict, optional
        Additional metadata to include in TIFF file
    compress : bool, optional
        Whether to use compression (default: True)
    
    Raises:
    -------
    ValueError
        If array is not a numpy ndarray or is empty
    IOError
        If file cannot be written
    """
    # Validate input array
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Input must be a numpy ndarray, got {type(array)}")
    
    if array.size == 0:
        raise ValueError("Cannot save empty array")
    
    # Convert to Path object and handle file extension
    filepath = Path(filepath)
    
    # Ensure .tiff extension
    if filepath.suffix.lower() not in ['.tiff', '.tif']:
        filepath = filepath.with_suffix('.tiff')
    
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare TIFF options
        tiff_options = {}
        
        if compress:
            tiff_options['compression'] = 'lzw'  # Lossless compression
        
        if metadata:
            tiff_options['metadata'] = metadata
        
        # Save the array
        tifffile.imwrite(filepath, array, **tiff_options)
        
        print(f"Successfully saved array to: {filepath}")
        print(f"  - Shape: {array.shape}")
        print(f"  - Data type: {array.dtype}")
        print(f"  - File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        raise IOError(f"Failed to save TIFF file '{filepath}': {e}") from e