import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict

def circular_gaussian_2d(coords: Tuple[np.ndarray, np.ndarray], 
                        amplitude: float, 
                        x_center: float, 
                        y_center: float, 
                        sigma: float, 
                        offset: float) -> np.ndarray:
    """
    2D Circular Gaussian function for diffraction pattern fitting
    
    Parameters:
    -----------
    coords : tuple of np.ndarray
        (x, y) coordinate grids
    amplitude : float
        Gaussian amplitude
    x_center, y_center : float
        Center coordinates
    sigma : float
        Gaussian width
    offset : float
        Background offset
    
    Returns:
    --------
    np.ndarray : Gaussian function values
    """
    x, y = coords
    r2 = (x - x_center)**2 + (y - y_center)**2
    return amplitude * np.exp(-r2 / (2 * sigma**2)) + offset

def fit_circular_gaussian_ring(data: np.ndarray,
                              initial_guess: Tuple[float, float, float, float, float],
                              inner_radius: float = 80,
                              outer_radius: float = 400) -> Dict:
    """
    Fit circular Gaussian to ring region with tight parameter bounds
    
    Parameters:
    -----------
    data : np.ndarray
        Input image data (1024, 1024)
    initial_guess : tuple
        Initial parameters [amplitude, x_center, y_center, sigma, offset]
    inner_radius : float
        Inner radius of fitting ring
    outer_radius : float
        Outer radius of fitting ring
    
    Returns:
    --------
    dict : Fitting results with parameters and uncertainties
    """
    # Unpack initial guess
    amp_guess, x_guess, y_guess, sigma_guess, offset_guess = initial_guess
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    
    # Create ring mask for fitting region
    r2_from_center = (x_coords - x_guess)**2 + (y_coords - y_guess)**2
    ring_mask = (r2_from_center >= inner_radius**2) & (r2_from_radius <= outer_radius**2)
    
    # Extract data points within ring
    x_fit = x_coords[ring_mask]
    y_fit = y_coords[ring_mask]
    z_fit = data[ring_mask]
    
    # Define parameter bounds based on expected deviations
    # amplitude: ±30%, center: ±30 pixels, sigma: ±20%, offset: ±50
    lower_bounds = [
        amp_guess * 0.7,           # amplitude -30%
        x_guess - 30,              # x_center -30 pixels
        y_guess - 30,              # y_center -30 pixels  
        sigma_guess * 0.8,         # sigma -20%
        offset_guess - 50          # offset -50
    ]
    
    upper_bounds = [
        amp_guess * 1.3,           # amplitude +30%
        x_guess + 30,              # x_center +30 pixels
        y_guess + 30,              # y_center +30 pixels
        sigma_guess * 1.2,         # sigma +20%
        offset_guess + 50          # offset +50
    ]
    
    # Ensure bounds are within physical limits
    lower_bounds[1] = max(0, lower_bounds[1])  # x_center min
    lower_bounds[2] = max(0, lower_bounds[2])  # y_center min
    upper_bounds[1] = min(data.shape[1], upper_bounds[1])  # x_center max
    upper_bounds[2] = min(data.shape[0], upper_bounds[2])  # y_center max
    lower_bounds[3] = max(1, lower_bounds[3])  # sigma min
    lower_bounds[4] = max(0, lower_bounds[4])  # offset min
    
    bounds = (lower_bounds, upper_bounds)
    
    try:
        # Perform fitting with tight bounds
        popt, pcov = curve_fit(circular_gaussian_2d, 
                              (x_fit, y_fit), z_fit, 
                              p0=initial_guess, 
                              bounds=bounds,
                              maxfev=2000)  # Reduced iterations due to tight bounds
        
        # Calculate parameter uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        result = {
            'amplitude': popt[0],
            'x_center': popt[1], 
            'y_center': popt[2],
            'sigma': popt[3],
            'offset': popt[4],
            'amplitude_err': perr[0],
            'x_center_err': perr[1],
            'y_center_err': perr[2],
            'sigma_err': perr[3],
            'offset_err': perr[4],
            'success': True,
            'ring_region': (inner_radius, outer_radius),
            'bounds_used': {
                'amplitude': (lower_bounds[0], upper_bounds[0]),
                'x_center': (lower_bounds[1], upper_bounds[1]),
                'y_center': (lower_bounds[2], upper_bounds[2]),
                'sigma': (lower_bounds[3], upper_bounds[3]),
                'offset': (lower_bounds[4], upper_bounds[4])
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        return {'success': False, 'error': str(e)}