import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def data_to_color_png(data_array: np.ndarray, filename: str = "data_color_map.png", cmap: str = 'viridis'):
    """
    Converts a 2D NumPy data array into a color-mapped PNG image for enhanced visualization.

    The data is normalized and colored using a specified Matplotlib colormap.

    Args:
        data_array (np.ndarray): The 2D data array (e.g., 1024x1024).
        filename (str): The complete file path and name for the output PNG image, 
                        including the target directory (e.g., 'output/data.png').
        cmap (str): The Matplotlib colormap name (e.g., 'viridis', 'jet', 'magma').
    """
    if data_array.ndim != 2:
        print("Error: Input data array must be 2-dimensional.")
        return

    print(f"Generating color map image '{filename}' using '{cmap}'...")

    # Normalize data to 0-1 range
    data_min = data_array.min()
    data_max = data_array.max()
    if data_min == data_max:
        # Avoid division by zero if array is constant
        data_normalized = np.zeros_like(data_array, dtype=float)
    else:
        data_normalized = (data_array - data_min) / (data_max - data_min)


    # Use Matplotlib to apply the colormap and convert to RGB
    mapper = plt.cm.get_cmap(cmap)
    color_mapped_data = mapper(data_normalized)

    # Convert the RGBA (0-1 float) array to an 8-bit RGB (0-255 uint8) array
    # We drop the alpha channel (the last dimension)
    img_array_8bit = (color_mapped_data[:, :, :3] * 255).astype(np.uint8)

    # Create PIL Image object and save it
    try:
        img = Image.fromarray(img_array_8bit, 'RGB')
        img.save(filename)
        print(f"Successfully saved color-mapped image to {filename}")
    except Exception as e:
        print(f"Error saving image to {filename}. Ensure the target directory exists. Error: {e}")


def bw_png_to_mask_ndarray(filename: str) -> np.ndarray:
    """
    Converts a black-and-white PNG image (intended as a mask) into a 
    NumPy array of 0s and 1s.

    - White pixels (intensity ~255) are converted to 1 (Signal/Keep).
    - Black pixels (intensity ~0) are converted to 0 (Unwanted/Discard).

    Args:
        filename (str): The complete file path and name of the black-and-white PNG mask,
                        including the source directory (e.g., 'input/mask.png').

    Returns:
        np.ndarray: The 2D integer NumPy mask array (0s and 1s).
    """
    print(f"Loading mask image '{filename}'...")
    try:
        # Open the image in grayscale ('L') to ensure 8-bit values (0-255)
        mask_img = Image.open(filename).convert('L')
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return np.array([])
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.array([])


    # Convert the image to a NumPy array (values 0-255)
    mask_array_raw = np.array(mask_img, dtype=np.uint8)

    # Convert 255 (white) to 1 and 0 (black) to 0.
    # We treat any value >= 128 as white (1) and anything below as black (0)
    manual_mask = (mask_array_raw >= 128).astype(int)

    print(f"Successfully converted mask to NumPy array of shape {manual_mask.shape} and dtype {manual_mask.dtype}")

    return manual_mask