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
    plt.show(block=False)

def plot_azimuthal_average(data, 
                              radius: float = 300,
                              num_bins: int = 300) -> None:
        """
        Plot the azimuthal average.
        """
        try:
            import matplotlib.pyplot as plt
            
            radii, intensities = data.azimuthal_average(radius, num_bins)
            
            plt.figure(figsize=(10, 6))
            plt.plot(radii, intensities, 'b-', linewidth=2, label='Azimuthal Average')
            
            plt.xlabel('Radial Distance (pixels)')
            plt.ylabel('Average Intensity')
            plt.title('Azimuthal Average Profile')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            
        except ImportError:
            print("Matplotlib not available for plotting")