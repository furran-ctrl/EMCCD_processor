import numpy as np
from pathlib import Path

initial_guess = (700,200)
delta = 5

x0 = np.array([initial_guess[0], initial_guess[1]])
        
print(x0)        
initial_simplex = np.array([
    x0,
    x0 + [delta, 0],
    x0 + [0, delta],
])
print(initial_simplex)

#import tiffile as tiff