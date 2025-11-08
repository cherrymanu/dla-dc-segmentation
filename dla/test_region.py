"""Quick test script for Region dataclass."""

from src.region import Region


def test_region():
    """Test Region dataclass functionality."""
    
    # Create regions
    r1 = Region(x=10, y=20, w=100, h=50, label="text")
    r2 = Region(x=120, y=20, w=100, h=50, label="text")
    r3 = Region(x=10, y=100, w=100, h=50, label="figure")
    
    print("=== Region Properties ===")
    print(f"Region 1: {r1}")
    print(f"  - x2: {r1.x2}, y2: {r1.y2}")
    print(f"  - Area: {r1.area}")
    print(f"  - Center: {r1.center}")
    print(f"  - Aspect ratio: {r1.aspect_ratio:.2f}")
    
    print("\n=== Overlap Tests ===")
    print(f"r1 overlaps r2 (no padding): {r1.overlaps(r2, padding=0)}")
    print(f"r1 overlaps r2 (padding=30): {r1.overlaps(r2, padding=30)}")
    print(f"r1 overlaps r3 (no padding): {r1.overlaps(r3, padding=0)}")
    
    print("\n=== Adjacency Tests ===")
    print(f"r1 is adjacent to r2: {r1.is_adjacent(r2, padding=30)}")
    print(f"r1 is adjacent to r3: {r1.is_adjacent(r3, padding=30)}")
    
    print("\n=== IoU Tests ===")
    r4 = Region(x=50, y=30, w=100, h=50, label="text")
    print(f"r1 IoU with r4 (overlapping): {r1.iou(r4):.3f}")
    print(f"r1 IoU with r2 (non-overlapping): {r1.iou(r2):.3f}")
    
    print("\n=== Alignment Tests ===")
    print(f"r1 vertically aligned with r3: {r1.is_vertically_aligned(r3)}")
    print(f"r1 horizontally aligned with r2: {r1.is_horizontally_aligned(r2)}")
    
    print("\n=== Merge Test ===")
    merged = Region.merge([r1, r2])
    print(f"Merged r1 and r2: {merged}")
    
    print("\n=== Dict Conversion ===")
    print(f"r1 as dict: {r1.to_dict()}")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    test_region()

