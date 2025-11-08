"""Test the complete segmentation pipeline."""

import cv2
import numpy as np

from src.main import segment_document


def visualize_final_segmentation(image: np.ndarray, regions, output_path: str):
    """Draw final segmentation with labels."""
    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Color mapping for labels
    label_colors = {
        "text": (0, 255, 0),      # Green
        "table": (255, 0, 0),     # Blue
        "figure": (0, 0, 255),    # Red
        "blank": (128, 128, 128)  # Gray
    }
    
    # Draw each region
    for i, region in enumerate(regions):
        color = label_colors.get(region.label, (255, 255, 255))
        
        # Draw thick rectangle
        cv2.rectangle(vis, (region.x, region.y), 
                     (region.x2, region.y2), color, 4)
        
        # Draw label text with background
        label_text = f"{i}: {region.label}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, 
                                                      font_scale, thickness)
        
        # Draw background rectangle for text
        padding = 5
        cv2.rectangle(vis, 
                     (region.x, region.y - text_h - baseline - padding * 2),
                     (region.x + text_w + padding * 2, region.y),
                     color, -1)
        
        # Draw text
        cv2.putText(vis, label_text, 
                   (region.x + padding, region.y - baseline - padding),
                   font, font_scale, (255, 255, 255), thickness)
    
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved final segmentation to {output_path}")


def test_complete_pipeline():
    """Test the complete document segmentation pipeline."""
    
    print("\n" + "=" * 70)
    print(" TESTING COMPLETE DOCUMENT LAYOUT SEGMENTATION PIPELINE")
    print("=" * 70 + "\n")
    
    # Run complete pipeline
    original, gray, binary, regions = segment_document(
        "inputs/academic.jpg",
        min_region=80,
        max_depth=15,
        do_merge=True
    )
    
    # Print detailed results
    print("\n" + "-" * 70)
    print("DETAILED REGION INFORMATION")
    print("-" * 70)
    
    for i, region in enumerate(regions):
        print(f"\nRegion {i}: {region.label.upper()}")
        print(f"  Position: ({region.x}, {region.y})")
        print(f"  Size: {region.w} x {region.h} pixels")
        print(f"  Area: {region.area:,} pixels")
        print(f"  Aspect ratio: {region.aspect_ratio:.2f}")
    
    # Visualize
    print("\n" + "-" * 70)
    print("VISUALIZATION")
    print("-" * 70)
    visualize_final_segmentation(original, regions, 
                                 "outputs/final_segmentation.jpg")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE TEST COMPLETE!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - outputs/final_segmentation.jpg (color-coded regions)")
    print("\nColor Legend:")
    print("  🟢 Green = Text")
    print("  🔵 Blue = Table")
    print("  🔴 Red = Figure")
    print("  ⚪ Gray = Blank")
    print()


if __name__ == "__main__":
    test_complete_pipeline()

