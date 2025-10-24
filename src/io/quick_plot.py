import matplotlib.pyplot as plt
import numpy as np

def plot_ndarray(Bkg: np.ndarray,Vmin,Vmax):

    plt.figure(figsize=(10, 8))
    # Create the plot
    im = plt.imshow(Bkg, cmap='viridis', aspect='equal',
        vmin=Vmin,  # <--- Manual minimum colorbar value
        vmax=Vmax ) 
    # Add colorbar
    plt.colorbar(im, label='Intensity')
    
    # Add labels and title
    plt.title('Plotted Image (1024*1024)')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
