import numpy as np
from typing import List
from pathlib import Path

from src.io.tiff_import import TiffLoaderBatch 

class CrudeBackground:
    #processing background images by computing median pixel values across multiple TIFF images
    
    def __init__(self, bkg_directory: str):
        """
        Initialize CrudeBackground with directory containing background TIFF images.
        
        Parameters:
        -----------
        bkg_directory : str
            Path to directory containing background TIFF images
        """
        self.bkg_directory = Path(bkg_directory)
        self.processed_bkg = None
        
        # Validate directory exists
        if not self.bkg_directory.exists():
            raise FileNotFoundError(f"Background directory not found: {self.bkg_directory}")
        if not self.bkg_directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.bkg_directory}")
    
    def process_background(self) -> np.ndarray:
        #Load all TIFF images in background directory and compute median for each pixel.

        print(f"Processing background images from: {self.bkg_directory}")
        
        # Load all TIFF images using TiffLoaderBatch
        try:
            images: List[np.ndarray] = TiffLoaderBatch(str(self.bkg_directory))
        except Exception as e:
            raise RuntimeError(f"Failed to load background images: {e}") from e
        if not images:
            raise ValueError(f"No TIFF images found in {self.bkg_directory}")
        
        print(f"Loaded {len(images)} background images")
        
        # Check that all images have the same shape
        first_shape = images[0].shape
        for i, img in enumerate(images):
            if img.shape != first_shape:
                raise ValueError(
                    f"Image shape mismatch: Image {i} has shape {img.shape}, "
                    f"expected {first_shape}. All background images must have identical dimensions."
                )
        
        # Convert list to numpy array stack
        image_stack = np.array(images)
        # Compute median along the stack axis (axis=0)
        self.processed_bkg = np.median(image_stack, axis=0)
        #self.processed_bkg = np.median(image_stack, axis=0)-np.mean(image_stack, axis=0)
        #print('Using!')
        print(f"Computed crude median background - Shape: {self.processed_bkg.shape}, ")
        
        return self.processed_bkg
    
    def get_processed_background(self) -> np.ndarray:
        #Get the processed background image. Processes if not already computed.

        if self.processed_bkg is None:
            return self.process_background()
        return self.processed_bkg