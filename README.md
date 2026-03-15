# 🚲 Beyond Proximity: Spatially Anisotropic Demand Dependencies in Urban Bike-Sharing

> **Weather-Augmented ConvLSTM Analysis of Washington DC Capital Bikeshare · January 2026**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/bikeshare_spatial_dependency.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Findings Journal](https://img.shields.io/badge/Published-Findings%20Journal-purple)](https://findingspress.org)

---

## 📌 Overview

This repository contains the full implementation of a **spatially-adapted permutation-based dependency analysis** framework for urban bike-sharing demand, applied to Washington DC Capital Bikeshare trip data from January 2026. It accompanies the paper:

> **"Beyond Proximity: Spatial Dependency in Washington DC Bike-Sharing — Weather-Augmented ConvLSTM Evidence"**  
> *Findings Journal — Transport Research (Submitted 2026)*

The core question: **do spatial dependencies in bike-sharing demand follow geographic proximity, or are they governed by functional urban linkages?** We answer this by training ConvLSTM models for member and casual users separately, augmenting them with real weather data, and computing a 64×64 permutation-based spatial dependency matrix Φ for each user type.

---

## 🎯 Aim & Scope

| | |
|---|---|
| **Study area** | Washington DC, USA |
| **Data source** | [Capital Bikeshare Open Data](https://capitalbikeshare.com/system-data) |
| **Period** | January 2026 (251,633 trips) |
| **Spatial resolution** | 8×8 regular grid (64 cells, ≈1,500 m each) |
| **Temporal resolution** | 30-minute intervals |
| **User types** | Member (205,329 trips) and Casual (46,304 trips) |
| **Weather source** | [Open-Meteo Historical Archive API](https://open-meteo.com/) |

### What this project establishes:
1. Whether spatial influence in bike-sharing conforms to the **geographic proximity assumption**
2. How **member and casual users differ** in their spatial dependency structures
3. Whether **weather augmentation** improves winter demand prediction beyond spatiotemporal models alone
4. A replication and extension of [Miao et al. (2025)](https://doi.org/10.1145/3748636.3763207) from NYC to a structurally distinct city

---

## 🔬 Key Findings

| Metric | Member | Casual |
|--------|--------|--------|
| **R²** | **0.816** | 0.427 |
| **MAE** (pickups/cell/interval) | 0.292 | 0.112 |
| **Mean Φ dependency** | 0.00819 | 0.00148 |
| **Max single-pair Φ** | 2.632 (cell 36→35) | 0.494 |
| **Proximity–dep. correlation** | −0.164 | −0.131 |
| **Anisotropy index** | **2.749** | 2.550 |
| **Member / Casual Φ ratio** | **5.52×** | — |

> 📍 **Spatial influence does NOT follow geographic proximity.** Dependencies are anisotropic, sparse, and concentrated in a downtown core cluster (cells 28, 29, 36, 37 — the Georgetown–Downtown–Capitol Hill corridor). Members exhibit 5.52× stronger inter-cell dependencies than casual users.

---

## 🗂️ Repository Structure

```
📦 bikeshare-spatial-dependency
 ┣ 📓 bikeshare_spatial_dependency.ipynb   ← Main Colab notebook (run this)
 ┣ 📄 README.md                            ← This file
 ┣ 📄 requirements.txt                     ← Python dependencies
 ┣ 📁 outputs/                             ← Generated figures and results
 ┃  ┣ 🖼️  activity_distribution.png
 ┃  ┣ 🖼️  spatial_dependencies_member.png
 ┃  ┣ 🖼️  spatial_dependencies_casual.png
 ┃  ┣ 🖼️  user_type_comparison.png
 ┃  ┣ 🖼️  training_history_member.png
 ┃  ┣ 🖼️  training_history_casual.png
 ┃  ┣ 🖼️  predictions_member.png
 ┃  ┣ 🖼️  predictions_casual.png
 ┃  ┣ 🖼️  grid_layout.png
 ┃  ┣ 🗺️  spatial_map_member.html
 ┃  ┣ 🗺️  spatial_map_casual.html
 ┃  ┣ 🔢  phi_matrix_member.npy
 ┃  ┣ 🔢  phi_matrix_casual.npy
 ┃  ┣ 📊  spatial_influence_analysis.csv
 ┃  ┗ 📝  summary_statistics.txt
 ┗ 📄 references_bikeshare.bib             ← BibTeX references
```

---

## ⚙️ How the Script Works

The pipeline runs in **18 sequential cells** in Google Colab:

```
 DATA               FEATURES              MODEL              ANALYSIS
  │                    │                    │                    │
  ▼                    ▼                    ▼                    ▼
Capital           Weather API          ConvLSTM           Permutation
Bikeshare    ──►  (Open-Meteo)   ──►  (3 layers)    ──►  Dependency
Trip CSV          + Temporal           + Dense             Matrix Φ
                    Features            Branch           (64×64 per
                  (7 inputs)          Separate            user type)
                                    Member/Casual
```

### Cell-by-cell breakdown

| Cell | Name | Description |
|------|------|-------------|
| 1 | **Install** | Installs TensorFlow, SHAP, Folium, GeoPandas, Plotly |
| 2 | **Imports** | Loads all libraries and sets random seeds (42) |
| 3 | **Config** | Defines study area bbox, grid size (8×8), time interval (30 min), model hyperparameters |
| 4 | **Data Extraction** | Extracts `202601-capitalbikeshare-tripdata.zip` to CSV |
| 5 | **Load & Clean** | Parses timestamps, removes invalid trips, filters to study area bbox |
| 6 | **Weather Data** | Fetches hourly temperature, precipitation, wind speed, humidity from Open-Meteo API for DC (38.9°N, 77.03°W); falls back to synthetic Jan DC weather if API unavailable |
| 7 | **Grid Creation** | Creates 8×8 regular lat/lon grid; assigns each trip a start and end cell ID |
| 8 | **Aggregation** | Aggregates pickups/dropoffs per cell per 30-min interval, split by member/casual |
| 9 | **Sequence Prep** | Builds 4-step lookback spatiotemporal tensors + 7-feature external vectors for each user type |
| 10 | **Model Build** | Constructs dual-input ConvLSTM: spatial branch (3 × ConvLSTM2D, 64 filters) + external dense branch (32→64 units), merged and output via 1×1 Conv |
| 11 | **Training** | Trains member and casual models with Adam (lr=0.001), MSE loss, early stopping (patience=10), LR reduction on plateau; 80/20 temporal split |
| 12 | **Φ Matrix** | For each of 64×63=4,032 source→target cell pairs: zeros out source cell history in 30 sampled test sequences, measures mean absolute change in target predictions → Φ(src→tgt) |
| 13 | **Dependency Viz** | Plots 64×64 Φ heatmap + net influence map with top-10 dependency arrows |
| 14 | **Interactive Map** | Generates Folium maps coloured by net spatial influence per cell |
| 15 | **User Comparison** | Side-by-side analysis: distribution histograms, proximity scatter, outward influence grids |
| 16 | **Predictions** | Visualises actual vs predicted pickup grids for 4 random test samples per user type |
| 17 | **Summary** | Prints all key statistics: R², MAE, anisotropy index, proximity correlation, dependency ratio |
| 18 | **Export** | Saves all outputs: `.npy` matrices, `.csv`, `.png` figures, `.html` maps |

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Upload `202601-capitalbikeshare-tripdata.zip` to `/content/` when prompted in Cell 4
3. Run all cells: **Runtime → Run all**
4. Download outputs from `/content/outputs/`

### Option B — Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Download data from Capital Bikeshare
# https://capitalbikeshare.com/system-data → January 2026
# Place the zip in the project root

# Launch Jupyter
jupyter notebook bikeshare_spatial_dependency.ipynb
```

### requirements.txt

```
tensorflow>=2.12
shap>=0.42
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
folium>=0.14
geopandas>=0.13
requests>=2.31
plotly>=5.15
```

---

## 📊 Selected Output Figures

<table>
<tr>
<td align="center"><b>Activity Distribution</b><br><sub>Member vs. Casual pickup heatmaps</sub></td>
<td align="center"><b>Member Dependency Matrix Φ</b><br><sub>64×64 permutation-based influence</sub></td>
</tr>
<tr>
<td align="center"><b>User Type Comparison</b><br><sub>Dependency distributions + proximity scatter</sub></td>
<td align="center"><b>Training Convergence</b><br><sub>MSE and MAE curves for both models</sub></td>
</tr>
</table>

> All figures are saved to `/content/outputs/` at 300 DPI during runtime.

---

## 🧠 Model Architecture

```
Spatial Input                    External Input
[batch, 4, 8, 8, 2]             [batch, 7]
        │                              │
   ConvLSTM2D (64)                Dense (32, ReLU)
   BatchNorm + Dropout(0.2)       BatchNorm
        │                         Dense (64, ReLU)
   ConvLSTM2D (64)                     │
   BatchNorm + Dropout(0.2)       Reshape (1,1,64)
        │                         UpSampling2D (8×8)
   ConvLSTM2D (64)                     │
   BatchNorm                           │
        └──────────┬────────────────────┘
                   │
              Concatenate
                   │
            Conv2D(2, 1×1, ReLU)
                   │
           Output [batch, 8, 8, 2]
           (pickups + dropoffs)
```

**Training configuration:**
- Optimizer: Adam (lr = 0.001, reduce on plateau: factor 0.5, patience 5)
- Loss: Mean Squared Error
- Early stopping: patience = 10, restore best weights
- Batch size: 32 | Max epochs: 30 | Split: 80/20 temporal

---

## 📐 Spatial Dependency Method

The Φ matrix is computed via **permutation-based perturbation** — a model-agnostic alternative to gradient-based SHAP:

```python
for each (source_cell, target_cell) pair:
    1. Take 30 random test sequences
    2. Zero out source_cell values in the lookback window
    3. Re-run model inference
    4. Φ(source → target) = mean |baseline_pred - perturbed_pred|
```

This yields a 64×64 directed influence matrix per user type. Key derived metrics:
- **Outward influence**: column sum of Φ — how much a cell drives others
- **Inward influence**: row sum of Φ — how much a cell is driven by others  
- **Net influence**: outward − inward
- **Anisotropy index**: CV of column-wise outward sums
- **Proximity correlation**: Pearson r(grid distance, Φ)

---

## 📚 Citation

If you use this code in your research, please cite:

KABIR, S. M. R. (2026). Weather-Enhanced-ConvLSTM-Modeling-of-Urban-Bike-Share-Demand-in-Washington-DC. Zenodo. https://doi.org/10.5281/zenodo.19028765

```bibtex
@misc{kabir_2026_19028765,
  author       = {KABIR, S M REDWAN},
  title        = {Weather-Enhanced-ConvLSTM-Modeling-of-Urban-Bike-
                   Share-Demand-in-Washington-DC
                  },
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19028765},
  url          = {https://doi.org/10.5281/zenodo.19028765},
}

**Key reference this work extends:**
```bibtex
@inproceedings{miao2025spatially,
  author    = {Miao, Congcong and He, Suining and Li, Yuyao and Zhang, Chuanrong},
  title     = {A spatially-adapted {SHAP} approach for interpreting deep bike usage
               learning and prediction},
  booktitle = {Proceedings of SIGSPATIAL '25},
  year      = {2025},
  doi       = {10.1145/3748636.3763207}
}
```


---

## 📄 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [Capital Bikeshare System Data](https://capitalbikeshare.com/system-data) | Trip-level records (January 2026) | Public |
| [Open-Meteo Historical Archive](https://open-meteo.com/en/docs/historical-weather-api) | Hourly weather for DC (38.9°N, 77.03°W) | Free API |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first. Areas for contribution:
- Extending to other cities or seasons
- Alternative dependency metrics (e.g., Granger causality)
- Dockless system adaptation
- Real-time inference pipeline

---

## 📝 License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.

---

<p align="center">
  <sub>Made with ☕ and 🚲 · Washington DC · 2026</sub>
</p>
