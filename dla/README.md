# Document Layout Analysis - Divide and Conquer

Implementation of a divide-and-conquer algorithm for document page segmentation with greedy region merging.

## Project Structure

```
dla/
├── src/               # Core algorithm only (preprocessing, xycut, labeling, merge, region)
├── experiments/       # run_experiments.py, runtime_analysis.py, plot_results.py
├── tests/             # Unit tests (test_pipeline, test_xycut, test_labeling, …)
├── inputs/            # Sample input images
├── outputs/           # Generated results (gitignored)
└── requirements.txt
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

Run from the `dla/` directory so that paths like `inputs/` resolve correctly.

```python
from src.main import segment_document

# Segment a document image (returns original, gray, binary, regions)
_, _, _, regions = segment_document('inputs/academic.jpg')
```

## Experiments

```bash
cd dla
python experiments/run_experiments.py
```

## Tests

```bash
cd dla
python tests/test_pipeline.py   # or test_xycut, test_labeling, test_preprocessing, test_region
```

## Algorithm Complexity

- Expected: O(N log N) where N = number of pixels
- Space: O(N) for image storage + O(k) for k regions

