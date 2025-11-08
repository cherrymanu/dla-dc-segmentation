"""Verify the binary PNG image is correct."""

import cv2
import numpy as np

# Check the new binary PNG
img = cv2.imread("outputs/academic_binary.png", cv2.IMREAD_GRAYSCALE)

print("=== Binary Image Verification ===")
print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")

unique_vals = np.unique(img)
print(f"Unique values: {len(unique_vals)}")
print(f"Values: {unique_vals}")

if len(unique_vals) == 2 and 0 in unique_vals and 255 in unique_vals:
    print("✓ Perfect binary image (only 0 and 255)")
else:
    print(f"⚠ Not perfectly binary, but close")

# Count pixels
zeros = np.sum(img == 0)
two_fifties = np.sum(img == 255)
total = img.size

print(f"\nPixel distribution:")
print(f"  Black (0): {zeros} pixels ({100*zeros/total:.1f}%)")
print(f"  White (255): {two_fifties} pixels ({100*two_fifties/total:.1f}%)")
print(f"  Other: {total - zeros - two_fifties} pixels ({100*(total-zeros-two_fifties)/total:.1f}%)")

print(f"\nContent density (dark/total): {zeros/total:.3f}")

