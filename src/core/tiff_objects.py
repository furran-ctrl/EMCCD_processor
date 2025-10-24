import numpy as np
from typing import Optional

class EMCCDimage:
    """
    EMCCD image class for storing and processing EMCCD camera data.
    """
    
    def __init__(self, raw_data: np.ndarray):
        """
        Initialize EMCCDimage with raw data.
        
        Parameters:
        -----------
        raw_data : np.ndarray
            Raw image data from EMCCD camera
        """
        self.raw_data = raw_data
        self.processed_data: Optional[np.ndarray] = None
    
    def remove_background(self, background: np.ndarray) -> None:
        """
        Subtract background from raw_data to produce processed_data.
        
        Parameters:
        -----------
        background : np.ndarray
            Background image to subtract from raw_data
        """
        self.processed_data = self.raw_data - background
    
    def get_processed_data(self) -> np.ndarray:
        """
        Get the processed data after background removal.
        
        Returns:
        --------
        np.ndarray
            Processed image data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call remove_background() first.")
        return self.processed_data