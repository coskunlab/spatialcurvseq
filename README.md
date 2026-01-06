# CurvSeq/CurvSee: Physics-Informed-Spatial-Curvature-Sequencing-in-Prostate-Cancer

This repository hosts the code, data processing scripts, and visualization tools developed for the **CurvSeq** project — a pipeline linking **glandular curvature**, **molecular expression**, **AFM-measured stiffness** in prostate adenocarcinoma tissues.
---

## 🔬 Overview

CurvSeq (short for *Curvature Sequencing*) provides a reproducible workflow to:
- Extract curvature metrics from segmented gland contours (QuPath, Cellpose-SAM, or manual masks)
- Visualize correlations between curvature and gene/protein data

The following images show the main pipeline of CurvSeq/CurvSee

![Alt text](pictures/Manusctipt_Figures_CurvSeq_up.png)

This schematic illustrates the overall CurvSeq/CurvSee workflow. Starting from multiplexed prostate tissue images (step 1), gland boundaries are extracted and used to compute local curvature profiles (step 2). These curvature values—distinguishing outward and inward regions—are then projected onto the corresponding molecular maps derived from spatial transcriptomics or proteomics data (step 3). Finally, curvature metrics are correlated with gene or protein expression levels to reveal spatial mechano-molecular relationships within prostate gland architecture (step 4).

![Alt text](pictures/Manusctipt_Figures_CurvSeq_down.png)

This figure demonstrates an example application of the CurvSeq workflow. The top panels show the computed curvature along a gland boundary (left) and the corresponding spatial distribution of FOXA1 expression (right). The bottom panels display microniche segmentation based on curvature using density based clustering, followed by correlation analysis between curvature clusters and FOXA1 levels. A positive Pearson correlation indicates that regions where the gland bends towards the outside are associated with elevated FOXA1 expression, highlighting spatial mechano-molecular coupling within the tissue.


## 📁 Repository Structure
This repository includes two Jupyter notebooks demonstrating the main steps of the CurvSeq pipeline:

-**Compute_curvature_contours.ipynb** — shows how to load gland boundary coordinates, compute local curvature values, and visualize curvature profiles along each contour.

-**Compute_plot_gene_curvature_correlation.ipynb** — demonstrates how to merge curvature data with molecular or protein expression (from AnnData files) and perform statistical correlation analysis (Pearson/Spearman), including visualizations of curvature–expression relationships.

## 💻 System Requirements

| Component | Requirement | Notes |
|------------|--------------|-------|
| **Operating System** | Windows 10 or Windows 11 | Tested primarily on Windows 11 (64-bit) |
| **Python Version** | 3.10.x | Works best with Conda or venv environments |
| **RAM** | ≥ 16 GB | Large TIFF/NumPy operations benefit from higher memory |
| **Storage** | ≥ 50 GB free | Required for AFM exports, TIFF masks, and processed datasets |
| **GPU (optional)** | NVIDIA CUDA-enabled | Recommended for Cellpose-SAM acceleration |
|  **Dependencies** | Listed in `environment.yml` | Includes `numpy`, `scikit-image`, `tifffile`, `napari`, `scanpy`, `hdbscan`, `matplotlib`, and others |

## ⚙️ Usage – Computing Curvature

The curvature computation module extracts local curvature values from a gland boundary represented as a 2D curve.

### Input Format
The pipeline expects a **2D NumPy array** of shape `(N, 2)` representing the coordinates of the gland boundary:
```python
import numpy as np

# Example: a gland contour stored as (y, x) or (row, col)
coords = np.load("path/to/gland_boundary.npy")  # shape (N, 2)
```
## AnnData Input
When integrating molecular or intensity data, the pipeline accepts an AnnData (.h5ad) file that stores spatial transcriptomics or imaging-derived intensity features for each observation.

```python
Typical format:
AnnData object with n_obs × n_vars = 71380 × 2
obs: 'X', 'Y',
```
Where X and Y are the cell coordinates, the vars are the genes of interest

## Test Data

The `test_data` folder contains:
- Contour coordinate files  
- Curvature values  
- An AnnData file with cell transcript counts and spatial coordinates
- 
To compute curvature from the contour coordinates, use the notebook:

- `Compute_curvature_contours.ipynb`

You can also compute curvature directly from the contour file using:

```python
curvature = compute_curvature_profile(
    contour_data,
    min_contour_length=20,
    window_size_ratio=20
)
```

After computing curvature, use the notebook:

Compute_plot_gene_curvature_correlation.ipynb

This notebook computes and visualizes the correlation between contour curvature
and the expression of a selected gene in cells located inside the contour.

## Python Packages

The following Python packages are required to run the CurvSeq pipeline:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import tifffile as tif
import scanpy as sc
import cv2
from scipy.ndimage import gaussian_filter1d
from hdbscan import HDBSCAN
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


