import numpy as np

with np.load('./datasets/real.npz', mmap_mode='r') as data:
    # Get a memory-mapped reference to just the 'images' array
    images = data['images']
    
    # Get the shape to understand the data structure
    print(f"Images shape: {images.shape}")