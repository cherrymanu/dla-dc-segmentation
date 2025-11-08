# Document / Page Layout Segmentation (Divide-and-Conquer)

## Overview
This project implements a divide-and-conquer (D&C) algorithm for page layout segmentation of documents (PDF scans, articles, forms). It recursively splits pages into regions using projection-profile and edge-density cues, and performs a bottom-up greedy merge of tiny adjacent regions. It outputs a hierarchical segmentation tree and labeled bounding boxes (text/table/figure/blank) using simple, interpretable heuristics.

## Components
- `code/segment.py`: Core D&C segmentation and region labeling.
- `code/generate_synthetic.py`: Synthetic multi-column page generator with ground truth.
- `code/evaluate.py`: Metrics (IoU, precision/recall) vs. ground truth.
- `code/run_experiments.py`: Runtime scaling and accuracy experiments; saves plots.
- `docs/paper/main.tex`: LaTeX paper scaffold (ACM-like) with filled sections.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
Run synthetic data experiments (generates pages, runs segmentation, evaluates, and plots):
```bash
python code/run_experiments.py --outdir outputs --num_pages 30 --resolutions 600 900 1200
```
Artifacts are written to `outputs/`:
- `seg_examples/` example overlays
- `plots/runtime_vs_pixels.png`
- `plots/accuracy_vs_resolution.png`

## Usage on your own images
```bash
python code/segment.py --image path/to/scan.png --save_overlay outputs/overlay.png --save_json outputs/regions.json
```

## Notes
- The D&C algorithm uses XY-cuts guided by horizontal/vertical projection valleys and edge density. Stopping criteria are feature-based (region size, variance, text-line periodicity estimate). Small neighboring regions are greedily merged.
- Labeling heuristic: text = high stroke density and line periodicity; table = grid-like structure (orthogonal edge peaks); figure = high edge variance, low periodicity; blank = low density.

## Reproducibility
- Random seeds are fixed by default; set `--seed` to control randomness.
- All code is self-contained; synthetic pages remove dataset dependency.

## License
For course use; feel free to adapt with citation.


