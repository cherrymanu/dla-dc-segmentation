# Document Layout Analysis - Divide and Conquer

Implementation of a divide-and-conquer algorithm for document page segmentation with greedy region merging.

## Project Structure

```
dla/
├── inputs/          # Input document images
├── src/             # Source code
├── outputs/         # Segmentation results and visualizations
│   └── plots/       # Runtime analysis plots
├── experiments/     # Runtime analysis scripts
└── requirements.txt # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Algorithm Overview

1. **Preprocessing**: Grayscale conversion and Otsu binarization
2. **Recursive XY-Cut**: Divide-and-conquer segmentation using projection profiles
3. **Region Labeling**: Classify regions as text, table, figure, or blank
4. **Greedy Merge**: Bottom-up merging of adjacent compatible regions

## Usage

```python
from src.main import segment_page

# Segment a document image
regions = segment_page('inputs/academic.jpg')
```

## Algorithm Complexity

- Expected: O(N log N) where N = number of pixels
- Space: O(N) for image storage + O(k) for k regions

