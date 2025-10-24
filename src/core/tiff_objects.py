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
    
    def find_diffraction_center(self, 
                           initial_guess: Tuple[float, float, float, float, float],
                           inner_radius: float = 80,
                           outer_radius: float = 400) -> Optional[Tuple[float, float]]:
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
        
            return self.diffraction_center
        else:
            print("Fitting failed")
            return None