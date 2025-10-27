import numpy as np
from typing import Optional, Tuple

from src.utils.gaussian_fitting import fit_circular_gaussian_ring

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
    
        print(f"Fitting circular Gaussian to ring region {inner_radius}-{outer_radius} pixels")
        print(f"Initial guess: A={initial_guess[0]:.1f}, center=({initial_guess[1]:.1f}, {initial_guess[2]:.1f}), "
            f"σ={initial_guess[3]:.1f}, offset={initial_guess[4]:.1f}")
    
        # Perform fitting
        fit_result = fit_circular_gaussian_ring(
            self.processed_data, initial_guess, inner_radius, outer_radius
        )
    
        if fit_result['success']:
            self.fit_result = fit_result
            self.diffraction_center = (fit_result['x_center'], fit_result['y_center'])
        
            print(f"Fitting successful!")
            print(f"Diffraction center: ({fit_result['x_center']:.2f} ± {fit_result['x_center_err']:.2f}, "
              f"{fit_result['y_center']:.2f} ± {fit_result['y_center_err']:.2f})")
            print(f"Gaussian width: {fit_result['sigma']:.2f} ± {fit_result['sigma_err']:.2f} pixels")
            print(f"Amp,offset: {fit_result['amplitude']:.2f} , {fit_result['offset']:.2f} pixels")
        
            return self.diffraction_center
        else:
            print("Fitting failed")
            return None
        
    def ring_centroid(self,
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
    # Pre-compute the ring mask for fitting 
    def iterative_ring_centroid(self,
                           initial_guess: Tuple[float, float],
                           max_iter: int = 10,
                           tolerance: float = 1.0) -> Tuple[float, float]:
        """
        Iteratively refine center using ring centroid method
        """
        current_center = initial_guess
        
        for iteration in range(max_iter):
            
            new_center = self.ring_centroid(current_center)
            # Check convergence
            shift_distance = np.sqrt((new_center[0] - current_center[0])**2 + 
                                    (new_center[1] - current_center[1])**2)
            print(f"Iteration {iteration+1}: Center shift = {shift_distance:.3f} pixels")
            #print(new_center,current_center)

            current_center = new_center
            if shift_distance < tolerance:
                break
        
        self.center_pos = self.ring_centroid(current_center,save_total_count=True)
        return self.center_pos
    
    def azimuthal_average(self, 
                         radius: float = 300,
                         num_bins: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate azimuthal average of the processed data around the center position.
        
        Parameters:
        -----------
        radius : float
            Maximum radius for azimuthal average (default: 300 pixels)
        num_bins : int
            Number of radial bins for the average
        
        Returns:
        --------
        tuple : (radial_positions, average_intensities)
            radial_positions : np.ndarray - Radial coordinates (pixels)
            average_intensities : np.ndarray - Azimuthally averaged intensities
        
        Raises:
        -------
        ValueError: If processed_data or center_pos is not set
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
