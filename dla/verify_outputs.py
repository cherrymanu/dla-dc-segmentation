"""Verify the generated images are correct."""

import cv2
import numpy as np


def check_image(path, expected_type):
    """Check if image matches expected type."""
    img = cv2.imread(path)
    if img is None:
        print(f"❌ {path} - Failed to load")
        return
    
    print(f"\n{path}:")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    
    if len(img.shape) == 2:
        print(f"  Channels: 1 (grayscale)")
        unique_vals = len(np.unique(img))
        print(f"  Unique values: {unique_vals}")
    else:
        print(f"  Channels: {img.shape[2]}")
        # Check if all channels are equal (grayscale saved as BGR)
        if img.shape[2] == 3:
            if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
                print(f"  Note: BGR format but all channels equal (grayscale)")
                unique_vals = len(np.unique(img[:,:,0]))
                print(f"  Unique values in channel: {unique_vals}")
            else:
                print(f"  Note: True color image (channels differ)")
    
    # Check value range
    print(f"  Min value: {img.min()}")
    print(f"  Max value: {img.max()}")
    print(f"  Mean value: {img.mean():.2f}")
    
    # For binary, check if only 0 and 255
    if expected_type == "binary":
        unique = np.unique(img)
        if len(unique) <= 2:
            print(f"  ✓ Binary image confirmed (only {len(unique)} unique values)")
        else:
            print(f"  ⚠ Not strictly binary ({len(unique)} unique values)")
            print(f"  Sample unique values: {unique[:10]}")

# Check all outputs
print("=== Verifying Generated Images ===")

check_image("outputs/academic_gray.jpg", "grayscale")
check_image("outputs/academic_binary.jpg", "binary")
check_image("outputs/preprocessing_comparison.jpg", "combined")
check_image("inputs/academic.jpg", "original")

