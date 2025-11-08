"""Debug valley detection to understand the recursion."""

from src.preprocessing import preprocess_image
from src.xycut import (compute_projection_profile, find_valleys,
                       valleys_to_split_indices)

# Load image
_, _, binary = preprocess_image("inputs/academic.jpg")
h, w = binary.shape

print(f"Image size: {w} x {h}\n")

# Test on full image
print("=== VERTICAL PROJECTION (Full Image) ===")
v_profile = compute_projection_profile(binary, axis=0)
v_valleys = find_valleys(v_profile, min_gap=5)
print(f"Valleys found: {len(v_valleys)}")
print(f"Valleys: {v_valleys}")

v_splits = valleys_to_split_indices(v_valleys, w, min_size=50)
print(f"Valid splits (min_size=50): {len(v_splits)}")
print(f"Splits: {v_splits}\n")

print("=== HORIZONTAL PROJECTION (Full Image) ===")
h_profile = compute_projection_profile(binary, axis=1)
h_valleys = find_valleys(h_profile, min_gap=5)
print(f"Valleys found: {len(h_valleys)}")
print(f"First 10 valleys: {h_valleys[:10]}")
print(f"Valley sizes: {[end-start for start, end in h_valleys[:10]]}")

h_splits = valleys_to_split_indices(h_valleys, h, min_size=50)
print(f"Valid splits (min_size=50): {len(h_splits)}")
print(f"First 10 splits: {h_splits[:10]}\n")

# Test on left column only (after first vertical split)
print("=== HORIZONTAL PROJECTION (Left Column Only) ===")
# Assuming left column is roughly x=129 to x=643
left_col = binary[:, 129:643]
h_profile_left = compute_projection_profile(left_col, axis=1)
h_valleys_left = find_valleys(h_profile_left, min_gap=5)
print(f"Valleys found: {len(h_valleys_left)}")
print(f"Valley sizes (first 20): {[end-start for start, end in h_valleys_left[:20]]}")

h_splits_left = valleys_to_split_indices(h_valleys_left, h, min_size=50)
print(f"Valid splits (min_size=50): {len(h_splits_left)}")

