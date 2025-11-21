import numpy as np
from typing import List
from pathlib import Path

from processor_code.io.tiff_import import TiffLoaderBatch 
from processor_code.core.tiff_objects import EMCCDimage

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
        self.processed_bkg = np.mean(image_stack, axis=0)
        #self.processed_bkg = np.median(image_stack, axis=0)-np.mean(image_stack, axis=0)
        #print('Using!')
        print(f"Computed crude median background - Shape: {self.processed_bkg.shape}, ")
        
        return self.processed_bkg
    
    def get_processed_background(self) -> np.ndarray:
        #Get the processed background image. Processes if not already computed.

        if self.processed_bkg is None:
            return self.process_background()
        return self.processed_bkg
    
class CalcBackground:
    """
    Process background images by computing mean pixel values across multiple TIFF images
    after removing X-ray spots from each individual image.
    """
    
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
    
    def process_background(self, 
                          chunk_size: int = 64,
                          sigma_threshold: float = 8.0,
                          beam_threshold: float = 5000) -> np.ndarray:
        """
        Load all TIFF images, remove X-ray spots from each, and compute mean background.
        
        Parameters:
        -----------
        chunk_size: Size of square chunks for X-ray filtering
        sigma_threshold: Number of MAD above median for X-ray detection  
        beam_threshold: Intensity threshold to identify beam regions
        
        Returns:
        --------
        np.ndarray: Processed background image (mean of cleaned images)
        """
        print(f"Processing background images from: {self.bkg_directory}")
        
        # Load all TIFF images
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
        
        # Process each image to remove X-ray spots
        cleaned_images = []
        
        for i, img_data in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} for X-ray removal...")
            
            # Create EMCCDimage instance
            img_obj = EMCCDimage(img_data)
            img_obj.copy_as_processed()

            # Apply X-ray filtering in place
            img_obj.filter_xray_spots_inplace(
                chunk_size=chunk_size,
                sigma_threshold=sigma_threshold,
                beam_threshold=beam_threshold
            )
            
            # Store the cleaned image (raw_data now contains NaN for removed pixels)
            cleaned_images.append(img_obj.processed_data)
        
        # Convert to numpy array stack
        image_stack = np.array(cleaned_images)
        
        # Compute mean along the stack axis, ignoring NaN values
        self.processed_bkg = np.nanmean(image_stack, axis=0)
        
        # Optional: Fill remaining NaN values with overall mean
        if np.any(np.isnan(self.processed_bkg)):
            overall_mean = np.nanmean(self.processed_bkg)
            self.processed_bkg = np.nan_to_num(self.processed_bkg, nan=overall_mean)
            print(f"Filled {np.sum(np.isnan(cleaned_images[0]))} NaN pixels with mean value")
        
        print(f"Computed X-ray cleaned mean background - Shape: {self.processed_bkg.shape}")
        print(f"Background value range: [{np.min(self.processed_bkg):.1f}, {np.max(self.processed_bkg):.1f}]")
        
        return self.processed_bkg