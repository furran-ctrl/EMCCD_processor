import sys
import os
import numpy as np
from typing import Dict, Tuple

# 手动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader

processed = EMCCDimage(TiffLoader(r'C:\Users\86177\Desktop\Diffraction_code\Temp_savespace',r'data_saved2.tiff'))
processed.copy_as_processed()

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
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

processed.iterative_ring_centroid([720,350])
plot_azimuthal_average(processed)
