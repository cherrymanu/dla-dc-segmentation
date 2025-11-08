"""
Runtime analysis experiments for document layout segmentation.
Measures runtime at different image sizes to verify O(N log N) complexity.
"""

import csv
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import segment_document


def resize_image(image_path: str, scale_factor: float) -> Tuple[np.ndarray, int, int]:
    """
    Load and resize image by scale factor.
    
    Args:
        image_path: Path to image
        scale_factor: Scale factor (e.g., 0.5 = half size)
        
    Returns:
        Tuple of (resized_image, width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    
    h, w = img.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, new_w, new_h


def measure_runtime(image_path: str, scale_factor: float, 
                    num_runs: int = 3) -> Tuple[float, int, int, int]:
    """
    Measure average runtime for segmentation at given scale.
    
    Args:
        image_path: Path to original image
        scale_factor: Scale factor for resizing
        num_runs: Number of runs to average
        
    Returns:
        Tuple of (avg_time_ms, width, height, num_regions)
    """
    # Resize and save temporary image
    temp_path = f"temp_scaled_{scale_factor:.2f}.jpg"
    
    resized, w, h = resize_image(image_path, scale_factor)
    cv2.imwrite(temp_path, resized)
    
    times = []
    num_regions = 0
    
    for run in range(num_runs):
        start_time = time.time()
        
        # Run segmentation
        _, _, _, regions = segment_document(temp_path, do_merge=True)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        num_regions = len(regions)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    avg_time = np.mean(times)
    
    return avg_time, w, h, num_regions


def run_runtime_experiments(image_path: str, 
                            scale_factors: List[float],
                            output_csv: str = "experiments/runtime_data.csv"):
    """
    Run runtime experiments at multiple scales.
    
    Args:
        image_path: Path to test image
        scale_factors: List of scale factors to test
        output_csv: Output CSV file path
    """
    print("=" * 70)
    print("RUNTIME ANALYSIS EXPERIMENTS")
    print("=" * 70)
    print(f"\nTest image: {image_path}")
    print(f"Scale factors: {scale_factors}")
    print(f"Runs per scale: 3 (for averaging)")
    print()
    
    results = []
    
    for i, scale in enumerate(scale_factors):
        print(f"\n[{i+1}/{len(scale_factors)}] Testing scale {scale:.2f}...")
        
        try:
            avg_time, w, h, num_regions = measure_runtime(image_path, scale, num_runs=3)
            pixels = w * h
            
            result = {
                'scale': scale,
                'width': w,
                'height': h,
                'pixels': pixels,
                'time_ms': avg_time,
                'num_regions': num_regions
            }
            
            results.append(result)
            
            print(f"  Size: {w} x {h} ({pixels:,} pixels)")
            print(f"  Time: {avg_time:.2f} ms")
            print(f"  Regions: {num_regions}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Save results
    if results:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['scale', 'width', 'height', 
                                                    'pixels', 'time_ms', 'num_regions'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to {output_csv}")
    
    print("\n" + "=" * 70)
    print("RUNTIME EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Test at multiple scales
    scale_factors = [
        0.3,   # 30% - small
        0.5,   # 50% - medium-small
        0.7,   # 70% - medium
        0.85,  # 85% - medium-large
        1.0,   # 100% - original
        1.15,  # 115% - large
        1.3    # 130% - very large
    ]
    
    results = run_runtime_experiments(
        "inputs/academic.jpg",
        scale_factors,
        "experiments/runtime_data.csv"
    )

