"""Test region labeling."""

import cv2
import numpy as np

from src.labeling import label_regions
from src.preprocessing import preprocess_image
from src.xycut import segment_page_xycut


def visualize_labeled_regions(image: np.ndarray, regions, output_path: str):
    """Draw bounding boxes with labels on image."""
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
        
        # Draw rectangle
        cv2.rectangle(vis, (region.x, region.y), 
                     (region.x2, region.y2), color, 3)
        
        # Draw label text
        label_text = f"{i}: {region.label}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label_text, 
                                               cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.6, 2)
        cv2.rectangle(vis, 
                     (region.x, region.y - text_h - 10),
                     (region.x + text_w + 5, region.y),
                     color, -1)
        
        # Text
        cv2.putText(vis, label_text, 
                   (region.x + 2, region.y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis)
    print(f"Saved labeled visualization to {output_path}")


def test_labeling():
    """Test labeling on segmented regions."""
    
    print("=== Testing Region Labeling ===\n")
    
    # Load and preprocess
    print("1. Preprocessing image...")
    original, gray, binary = preprocess_image("inputs/academic.jpg")
    
    # Run XY-Cut
    print("\n2. Running XY-Cut segmentation...")
    regions = segment_page_xycut(binary)
    
    # Label regions
    print("\n3. Labeling regions...")
    labeled_regions = label_regions(gray, binary, regions)
    
    # Print results
    print(f"\n4. Labeled regions:")
    for i, region in enumerate(labeled_regions):
        print(f"   Region {i}: {region.label:6s} - {region}")
    
    # Visualize
    print(f"\n5. Creating visualization...")
    visualize_labeled_regions(original, labeled_regions, 
                              "outputs/labeled_result.jpg")
    
    print("\n✓ Labeling test completed!")
    print("Check outputs/labeled_result.jpg to see the labeled regions")
    print("  Green = text, Blue = table, Red = figure, Gray = blank")


if __name__ == "__main__":
    test_labeling()

