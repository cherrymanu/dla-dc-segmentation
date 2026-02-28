# Divide-and-Conquer Document Layout Segmentation

A modular implementation of a divide-and-conquer algorithm for document layout analysis using recursive XY-cut segmentation with feature-guided splitting and post-processing region merging.

## 📋 Overview

This project implements a divide-and-conquer approach to segment document images into labeled regions (text, table, figure, blank). The algorithm recursively partitions pages using projection profiles, edge density, and periodicity features, achieving O(N log N) time complexity.

## 🏗️ Project Structure

```
.
├── dla/                      # Main implementation
│   ├── src/                  # Core algorithm (library code only)
│   │   ├── main.py           # Pipeline orchestrator
│   │   ├── preprocessing.py  # Grayscale, Otsu binarization
│   │   ├── xycut.py          # Recursive XY-cut segmentation
│   │   ├── labeling.py       # Region classification
│   │   ├── merge.py          # Greedy merging (post-processing)
│   │   └── region.py         # Region dataclass
│   ├── experiments/          # run_experiments.py, runtime_analysis, plot_results
│   ├── tests/                # Unit tests (test_pipeline, test_xycut, …)
│   ├── inputs/               # Sample images
│   ├── outputs/              # Results (gitignored)
│   └── requirements.txt
├── code/                     # Shared utilities (used by dla)
│   ├── evaluate.py          # IoU, precision, recall, F1
│   └── generate_synthetic.py # Synthetic document generator
├── docs/paper/               # Paper
│   ├── main.tex
│   ├── refs.bib
│   └── *.png                 # Paper figures
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
# or for dla/ specific requirements:
pip install -r dla/requirements.txt
```

### Basic Usage

```python
from dla.src.main import segment_document

# Segment a document image
original, gray, binary, regions = segment_document(
    "path/to/image.jpg",
    min_region=80,
    max_depth=15,
    do_merge=True
)

# regions is a list of Region objects with:
# - x, y, w, h: bounding box coordinates
# - label: "text", "table", "figure", or "blank"
```

### Running Experiments

```bash
cd dla
python experiments/run_experiments.py
```

This will:
- Generate synthetic data at multiple resolutions
- Run segmentation and evaluation
- Generate runtime vs pixels and accuracy vs resolution plots

## 🔬 Algorithm

The algorithm consists of four main phases:

1. **Recursive XY-Cut Splitting**: Recursively partitions regions based on projection profiles (whitespace valleys) and feature homogeneity
2. **Region Labeling**: Classifies regions using edge density, content density, and FFT-based periodicity
3. **Boundary Refinement**: Refines region boundaries to better align with content
4. **Post-Processing Merge**: Greedily merges adjacent compatible regions

### Complexity

- **Time Complexity**: O(N log N) where N is the number of pixels
- **Space Complexity**: O(N) for the recursion stack and region storage

## 📊 Results

- **Runtime**: Verified O(N log N) scaling with R² > 0.96
- **Accuracy**: Stable F1 scores across different image resolutions
- **Plots**: See `docs/paper/` for runtime and accuracy visualizations

## 📄 Paper

The complete LaTeX paper with algorithm descriptions, proofs of correctness, and experimental results is available in `docs/paper/main.tex`.

## 🧪 Testing

```bash
cd dla
python tests/test_pipeline.py      # Test full pipeline
python tests/test_xycut.py         # Test XY-cut algorithm
python tests/test_labeling.py      # Test region labeling
python tests/test_preprocessing.py # Test image preprocessing
python tests/test_region.py        # Test Region dataclass
```

## 📝 Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- SciPy

See `requirements.txt` for full list.


The **greedy merge algorithm** (`dla/src/merge.py`) and its integration in this pipeline were implemented independently. The divide-and-conquer XY-cut and labeling components were developed as part of a team project.

## 📚 References

See `docs/paper/refs.bib` for the bibliography.
