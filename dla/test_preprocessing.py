"""Test script for preprocessing utilities."""

import os

from src.preprocessing import (get_content_density, load_image, otsu_binarize,
                               preprocess_image, save_image, to_grayscale,
                               visualize_preprocessing)


def test_preprocessing():
    """Test preprocessing pipeline on academic.jpg."""
    
    print("=== Testing Preprocessing Pipeline ===\n")
    
    # Input and output paths
    input_path = "inputs/academic.jpg"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test complete pipeline
    print("1. Running complete preprocessing pipeline...")
    original, gray, binary = preprocess_image(input_path)
    
    print(f"   - Original shape: {original.shape}")
    print(f"   - Gray shape: {gray.shape}")
    print(f"   - Binary shape: {binary.shape}")
    
    # Test content density
    print("\n2. Calculating content density...")
    full_density = get_content_density(binary)
    print(f"   - Full image content density: {full_density:.3f}")
    
    # Test on specific ROI (top-left corner)
    roi_density = get_content_density(binary, roi=(0, 0, 200, 200))
    print(f"   - ROI (0,0,200,200) content density: {roi_density:.3f}")
    
    # Save intermediate results
    print("\n3. Saving intermediate results...")
    save_image(gray, os.path.join(output_dir, "academic_gray.jpg"))
    print(f"   - Saved grayscale: outputs/academic_gray.jpg")
    
    save_image(binary, os.path.join(output_dir, "academic_binary.png"), is_binary=True)
    print(f"   - Saved binary: outputs/academic_binary.png (PNG to preserve binary values)")
    
    # Create visualization
    print("\n4. Creating visualization...")
    viz = visualize_preprocessing(original, gray, binary)
    save_image(viz, os.path.join(output_dir, "preprocessing_comparison.jpg"))
    print(f"   - Saved comparison: outputs/preprocessing_comparison.jpg")
    
    print("\n✓ All preprocessing tests completed successfully!")
    print(f"\nCheck the outputs/ directory for results.")


if __name__ == "__main__":
    test_preprocessing()

