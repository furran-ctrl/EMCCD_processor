import numpy as np
from typing import Optional, Tuple

from src.utils.gaussian_fitting import fit_circular_gaussian_ring
from src.core.mask_class import RadialMasks, RingMask

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
        self.center_pos: Optional[Tuple] = None
        self.total_count: Optional[float] = None
    
    def remove_background(self, background: np.ndarray) -> None:
        """
        Subtract background from raw_data to produce processed_data.
        
        Parameters:
        -----------
        background : np.ndarray
            Background image to subtract from raw_data
        """
        self.processed_data = self.raw_data - background
        #(MAYBE REMOVE GENERAL SHIFTS VIA TUNING)
        #(implement later: calculate mean as bkg but filter with sigma)
    
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
    
    def copy_as_processed(self):
        #DEBUG ONLY!!
        self.processed_data = self.raw_data
    
    def find_diffraction_center(self, 
                           initial_guess: Tuple[float, float, float, float, float],
                           inner_radius: float = 40,
                           outer_radius: float = 200) -> Optional[Tuple[float, float]]:
        '''
        Find diffraction center using circular Gaussian fitting with tight bounds
        
        Parameters:
        -----------
        initial_guess : tuple
            [amplitude, x_center, y_center, sigma, offset]
        inner_radius : float
            Inner radius of fitting ring
        outer_radius : float
            Outer radius of fitting ring
        
        Returns:
        --------
        tuple or None : Found center coordinates (x, y)
        '''
        if self.processed_data is None:
            raise ValueError("Call remove_background() first")
    
        '''print(f"Fitting circular Gaussian to ring region {inner_radius}-{outer_radius} pixels")
        print(f"Initial guess: A={initial_guess[0]:.1f}, center=({initial_guess[1]:.1f}, {initial_guess[2]:.1f}), "
            f"σ={initial_guess[3]:.1f}, offset={initial_guess[4]:.1f}")'''
    
        # Perform fitting
        fit_result = fit_circular_gaussian_ring(
            self.processed_data, initial_guess, inner_radius, outer_radius
        )
    
        if fit_result['success']:
            #self.fit_result = fit_result
            #self.diffraction_center = (fit_result['x_center'], fit_result['y_center'])
            self.center_pos = (fit_result['x_center'], fit_result['y_center'])

            '''print(f"Fitting successful!")
            print(f"Diffraction center: ({fit_result['x_center']:.2f} ± {fit_result['x_center_err']:.2f}, "
              f"{fit_result['y_center']:.2f} ± {fit_result['y_center_err']:.2f})")
            print(f"Gaussian width: {fit_result['sigma']:.2f} ± {fit_result['sigma_err']:.2f} pixels")
            print(f"Amp,offset: {fit_result['amplitude']:.2f} , {fit_result['offset']:.2f} pixels")'''

            y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
            # Create ring mask
            distances = np.sqrt((x_coords - initial_guess[1])**2 + (y_coords - initial_guess[2])**2)
            ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
            
            # Extract ring region
            ring_data = self.processed_data[ring_mask]
        
            # Calculate total intensity
            self.total_intensity = np.sum(ring_data)
            return self.center_pos
        else:
            print("Fitting failed")
            return None

    def ring_centroid(self,
                          ring_mask: RingMask,
                          center_guess: Tuple[float, float],
                          save_total_count: bool = False) -> Tuple[float, float]:
        """
        Calculate centroid using precomputed ring mask and specified center guess.
        
        This optimized version shifts the image data to align the specified center
        guess with the precomputed mask center, eliminating the need to recalculate
        the ring mask for each image and center combination.
        
        Args:
            ring_mask: Precomputed RingMask object centered at image center
            center_guess: Center coordinates to use for centroid calculation (x, y)
            save_total_count: If True, save total intensity count to self.total_count
            
        Returns:
            tuple: Centroid coordinates (x_center, y_center)
            
        Raises:
            ValueError: If processed_data is not available
            ValueError: If image shape doesn't match mask shape
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        # Validate image shape matches mask shape
        if self.processed_data.shape != ring_mask.image_shape:
            raise ValueError(
                f"Image shape {self.processed_data.shape} doesn't match "
                f"mask shape {ring_mask.image_shape}"
            )
        
        # Calculate shift needed to align specified center with mask center
        center_x, center_y = center_guess
        mask_center_x, mask_center_y = (
            ring_mask.image_shape[1] // 2, 
            ring_mask.image_shape[0] // 2
        )
        
        shift_x = int(round(center_x - mask_center_x))
        shift_y = int(round(center_y - mask_center_y))
        
        # Shift image data to align specified center with precomputed mask center
        shifted_data = np.roll(self.processed_data, shift=(-shift_y, -shift_x), axis=(0, 1))
        
        # Extract intensity values within ring region using precomputed mask
        ring_data = shifted_data[ring_mask.mask]
        
        # Calculate weighted centroid using precomputed coordinates
        total_intensity = np.sum(ring_data)
        
        if total_intensity > 0:
            x_center = np.sum(ring_mask.x_coords_ring * ring_data) / total_intensity
            y_center = np.sum(ring_mask.y_coords_ring * ring_data) / total_intensity
        else:
            # Fallback to center guess if no intensity in ring
            x_center, y_center = center_guess
        
        # Apply reverse shift to get coordinates in original image space
        final_x_center = x_center + shift_x
        final_y_center = y_center + shift_y
        
        if save_total_count:
            self.total_count = total_intensity
        
        return final_x_center, final_y_center

    def ring_centroid_legacy(self,
                  center_guess: Tuple[float, float],
                  inner_radius: float = 40,
                  outer_radius: float = 200,
                  save_total_count: bool = False) -> Tuple[float, float]:
        """
        Calculate centroid within ring region
        """
        y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
        # Create ring mask
        distances = np.sqrt((x_coords - center_guess[0])**2 + (y_coords - center_guess[1])**2)
        ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
        
        # Extract ring region
        ring_data = self.processed_data[ring_mask]
        ring_x = x_coords[ring_mask]
        ring_y = y_coords[ring_mask]
    
        # Calculate weighted centroid
        total_intensity = np.sum(ring_data)
        if total_intensity > 0:
            x_center = np.sum(ring_x * ring_data) / total_intensity
            y_center = np.sum(ring_y * ring_data) / total_intensity
        else:
            x_center, y_center = center_guess
        
        if save_total_count:
            self.total_count = total_intensity
        return x_center, y_center
 
    def iterative_ring_centroid(self,
                           ring_mask: RingMask,
                           initial_guess: Tuple[float, float],
                           max_iter: int = 10,
                           tolerance: float = 1.0) -> Tuple[float, float]:
        """
        Iteratively refine center coordinates using ring centroid method.
        
        This method performs multiple iterations of ring centroid calculation,
        using the result from each iteration as the center for the next ring.
        The process continues until convergence or maximum iterations reached.
        
        Args:
            initial_guess: Initial center coordinates as (x, y) tuple in pixels.
            max_iter: Maximum number of iterations to perform. Defaults to 10.
            tolerance: Convergence tolerance in pixels. Iteration stops when center 
                      shift is less than this value. Defaults to 1.0.
        
        Returns:
            Final refined center coordinates as (x, y) tuple.

        Note:
            The ring centroid method calculates the intensity-weighted center of mass
            within a ring region around the current center estimate. This iterative
            approach helps converge to the true diffraction pattern center.
        """
        current_center = initial_guess
        
        for iteration in range(max_iter):
            
            new_center = self.ring_centroid(ring_mask, current_center)
            # Check convergence
            shift_distance = np.sqrt((new_center[0] - current_center[0])**2 + 
                                    (new_center[1] - current_center[1])**2)
            #print(f"Iteration {iteration+1}: Center shift = {shift_distance:.3f} pixels")

            current_center = new_center
            if shift_distance < tolerance:
                break
        
        self.center_pos = self.ring_centroid(ring_mask, current_center, save_total_count=True)
        return self.center_pos
    
    def azimuthal_average(self, 
                                  radial_masks: RadialMasks,
                                  radius: float = 300,
                                  num_bins: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate azimuthal average using precomputed radial masks.
        
        This optimized version shifts the image data to align with the precomputed
        masks instead of recalculating masks for each image. This provides significant
        performance improvement when processing multiple images with the same dimensions.
        
        Args:
            radial_masks: Precomputed RadialMasks object
            radius: Maximum radius for azimuthal average (must match radial_masks.radius)
            num_bins: Number of radial bins (must match radial_masks.num_bins)
            
        Returns:
            tuple: Contains two arrays:
                - radial_positions: Bin center coordinates (num_bins,)
                - average_intensities: Azimuthally averaged intensities (num_bins,)
                
        Raises:
            ValueError: If processed_data or center_pos is not set
            ValueError: If image shape doesn't match mask shape
            ValueError: If radius or num_bins don't match precomputed masks
            
        Example:
            >>> # Precompute masks once
            >>> masks = precompute_radial_masks((1024, 1024))
            >>> 
            >>> # Process multiple images efficiently
            >>> for image in images:
            >>>     centers, intensities = image.azimuthal_average_optimized(masks)
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        if self.center_pos is None:
            raise ValueError("Center position not set. Call find_diffraction_center() first.")
        
        # Validate input parameters match precomputed masks
        if self.processed_data.shape != radial_masks.image_shape:
            raise ValueError(
                f"Image shape {self.processed_data.shape} doesn't match "
                f"mask shape {radial_masks.image_shape}"
            )
        
        if radius != radial_masks.radius:
            raise ValueError(
                f"Requested radius {radius} doesn't match precomputed radius {radial_masks.radius}"
            )
        
        if num_bins != radial_masks.num_bins:
            raise ValueError(
                f"Requested num_bins {num_bins} doesn't match precomputed num_bins {radial_masks.num_bins}"
            )
        
        # Calculate shift needed to align image center with mask center
        center_x, center_y = self.center_pos
        mask_center_x, mask_center_y = (
            radial_masks.image_shape[1] // 2, 
            radial_masks.image_shape[0] // 2
        )
        
        shift_x = int(round(center_x - mask_center_x))
        shift_y = int(round(center_y - mask_center_y))
        
        # Shift image data to align with precomputed mask center
        shifted_data = np.roll(self.processed_data, shift=(-shift_y, -shift_x), axis=(0, 1))
        
        # Calculate azimuthal average using precomputed masks
        radial_average = np.zeros(num_bins)
        pixel_counts = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = radial_masks.masks[i]
            if np.any(mask):
                radial_average[i] = np.mean(shifted_data[mask])
                pixel_counts[i] = np.sum(mask)
        
        return radial_masks.bin_centers, radial_average
    
    def azimuthal_average_legacy(self, 
                               radius: float = 300,
                               num_bins: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy implementation of azimuthal average (original version).
        
        This is kept for backward compatibility and for cases where
        precomputed masks are not available.
        
        Args:
            radius: Maximum radius for azimuthal average in pixels
            num_bins: Number of radial bins
            
        Returns:
            tuple: (radial_positions, average_intensities)
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        if self.center_pos is None:
            raise ValueError("Center position not set. Call find_diffraction_center() first.")
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
        # Calculate distances from center
        center_x, center_y = self.center_pos
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Create radial bins
        bin_edges = np.linspace(0, radius, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize arrays for results
        radial_average = np.zeros(num_bins)
        pixel_counts = np.zeros(num_bins)
        
        # Calculate azimuthal average
        for i in range(num_bins):
            # Create mask for current radial bin
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            
            if np.any(mask):
                radial_average[i] = np.mean(self.processed_data[mask])
                pixel_counts[i] = np.sum(mask)
        
        return bin_centers, radial_average
