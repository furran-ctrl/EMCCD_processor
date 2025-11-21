import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class RadialMasks:
    """
    Precomputed radial masks for azimuthal averaging.
    
    Attributes:
        bin_centers: Center coordinates of radial bins (num_bins,)
        masks: List of boolean masks for each radial bin
        image_shape: Original image shape the masks were computed for
        radius: Maximum radius used for mask computation
        num_bins: Number of radial bins
    """
    bin_centers: np.ndarray
    masks: List[np.ndarray]
    image_shape: Tuple[int, int]
    radius: float
    num_bins: int

def precompute_radial_masks(image_shape: Tuple[int, int] = (1024, 1024), 
                           radius: float = 512, 
                           num_bins: int = 512) -> RadialMasks:
    """
    Precompute radial masks for azimuthal averaging.
    
    This function creates a set of boolean masks for radial bins centered
    at the image center. These masks can be reused for multiple images
    by simply shifting the image data to align with the mask center.
    
    Args:
        image_shape: Shape of the input images (height, width)
        radius: Maximum radius for azimuthal average in pixels
        num_bins: Number of radial bins
        
    Returns:
        RadialMasks object containing precomputed masks and bin information
        
    Example:
        >>> masks = precompute_radial_masks((1024, 1024), radius=512, num_bins=512)
        >>> print(f"Created {len(masks.masks)} radial masks")
    """
    height, width = image_shape
    
    # Create coordinate grid centered at image center
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distances from image center
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create radial bins
    bin_edges = np.linspace(0, radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Precompute masks for each radial bin
    masks = []
    for i in range(num_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        masks.append(mask)
    
    return RadialMasks(
        bin_centers=bin_centers,
        masks=masks,
        image_shape=image_shape,
        radius=radius,
        num_bins=num_bins
    )

def precompute_azimuthal_average_masks(radial_masks):
    """
    Precompute optimized data structures for fast azimuthal averaging.
    
    Args:
        radial_masks: Precomputed RadialMasks object
        image_shape: Shape of the images to be processed
        
    Returns:
        dict: Precomputed data structures for fast averaging
    """
    num_bins = radial_masks.num_bins
    image_shape = radial_masks.image_shape

    # Precompute flat indices for all masks
    flat_indices = []
    valid_mask_flags = []
    
    for i in range(num_bins):
        mask = radial_masks.masks[i]
        if np.any(mask):
            # Convert 2D boolean mask to flat indices
            indices = np.where(mask.ravel())[0]
            flat_indices.append(indices)
            valid_mask_flags.append(True)
        else:
            flat_indices.append(np.array([], dtype=int))
            valid_mask_flags.append(False)
    
    # Precompute which bins have valid data
    valid_bins = np.array([i for i in range(num_bins) if valid_mask_flags[i]])
    
    return {
        'flat_indices': flat_indices,
        'valid_mask_flags': valid_mask_flags,
        'valid_bins': valid_bins,
        'num_bins': num_bins,
        'total_pixels': image_shape[0] * image_shape[1]
    }

@dataclass
class RingMask:
    """
    Precomputed ring mask for centroid calculation.
    
    Attributes:
        mask: Boolean mask defining the ring region centered at image center
        x_coords_ring: X coordinates within the ring region  
        y_coords_ring: Y coordinates within the ring region
        image_shape: Original image shape the mask was computed for
        inner_radius: Inner radius of the ring
        outer_radius: Outer radius of the ring
    """
    mask: np.ndarray
    x_coords_ring: np.ndarray
    y_coords_ring: np.ndarray
    image_shape: Tuple[int, int]
    inner_radius: float
    outer_radius: float

def precompute_ring_mask(image_shape: Tuple[int, int] = (1024, 1024),
                        inner_radius: float = 40,
                        outer_radius: float = 200) -> RingMask:
    """
    Precompute ring mask centered at image center for centroid calculation.
    
    This function creates a ring-shaped boolean mask centered at the image center
    and extracts the coordinates within the ring. The precomputed mask can be 
    reused for multiple images by shifting the image data to align different centers
    with the precomputed mask.
    
    Args:
        image_shape: Shape of the input images (height, width)
        inner_radius: Inner radius of the ring in pixels
        outer_radius: Outer radius of the ring in pixels
        
    Returns:
        RingMask object containing precomputed mask and coordinate information
    """
    height, width = image_shape
    
    # Create coordinate grid centered at image center
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distances from image center
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create ring mask centered at image center
    ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
    
    # Extract coordinates within ring region
    x_coords_ring = x_coords[ring_mask]
    y_coords_ring = y_coords[ring_mask]
    
    return RingMask(
        mask=ring_mask,
        x_coords_ring=x_coords_ring,
        y_coords_ring=y_coords_ring,
        image_shape=image_shape,
        inner_radius=inner_radius,
        outer_radius=outer_radius
    )
