<div align="center">

# 🌐 Regional Search Interest Explorer
### *Unsupervised Machine Learning & Geospatial Policy Intelligence Platform*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Decode regional search intent, segment behavioral audience personas, and convert unsupervised machine learning into actionable public-policy directives and commercial playbooks.</b>
</p>

[Key Features](#-key-features) • [Dashboard Tour](#-dashboard-tour) • [Quickstart](#-quickstart) • [Methodology](#-methodology) • [Data Formats](#-supported-data-formats)

---

</div>

## 📌 Overview

Public search behavior on **Google Trends** reveals underlying regional priorities, socio-economic demand, and awareness disparities. However, raw search trends are relative, fragmented, and difficult for non-technical stakeholders to interpret.

The **Regional Search Interest Explorer** bridges advanced unsupervised machine learning with real-world decision-making. It enables researchers, policy analysts, and business leaders to:
- Group regions into statistically validated behavioral clusters across multi-topic interest profiles (*Education, Jobs, Agriculture, Politics, Sports, Tourism*).
- Interact with an executive **India Infographic Vector Map** with real administrative boundaries and rich hover dossiers.
- Simulate policy interventions using the **Strategic Priority Matrix** (Tier 1 High Priority to Tier 3 Baseline).
- Generate **AI-ready executive briefings** with one-click prompts for ChatGPT, Gemini, and Claude.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🗺️ **India Vector Infographic Map** | Stand-alone administrative map of all 36 Indian states and UTs with crisp state boundaries, cluster fills, interactive hovercards, and donut distribution charts. |
| 🎯 **Dual Clustering Algorithms** | **K-Medians (Manhattan distance)** for outlier-resilient median centroids, plus standard **K-Means (Euclidean)**. |
| 💼 **Strategic Action Matrix** | Automated regional prioritization ranking (Tiers 1–3) with customized public-policy initiatives and commercial GTM playbooks. |
| 🤖 **AI Executive Briefing Assistant** | One-click prompt synthesizer translating active cluster centroids and priority indices into tailored memos for leading LLMs. |
| 📈 **2D Behavioral Landscape (PCA)** | Dimensionality reduction projecting complex multi-topic profiles onto an intuitive 2D persona landscape. |
| 🧹 **Zero-Config Data Cleaning** | Automatically cleans raw Google Trends exports, stripping metadata headers, handling `<1` values, timestamps, and percentages. |
| 🩺 **Executive Health Diagnostics** | Plain-language quality scorecards translating Silhouette and Davies–Bouldin scores into actionable clarity assessments. |

---

## 🖥️ Dashboard Tour

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🌐 Regional Search Interest Explorer                                       │
├─────────────┬───────────────────────────────────────────────────────────────┤
│ ⚙️ Sidebar  │ [0. Overview]          [1. Regional Explorer] [2. Profiles]  │
│ - Upload CSV│ [3. Strategic Matrix]  [4. 🤖 AI Briefing]   [5. Regional Map]│
│ - Topics    │ [6. Model Diagnostics] [7. About & Limitations]               │
│ - Algorithm │                                                               │
│ - K Sliders ├───────────────────────────────────────────────────────────────┤
│ - Advanced  │ 🇮🇳 India Infographic Map & Real-Time Regional Analytics      │
└─────────────┴───────────────────────────────────────────────────────────────┘
```

1. **🏠 Overview:** Executive summary scorecards, regional distribution bar charts, and sortable cluster assignment tables.
2. **🔍 Regional Explorer:** 2D interactive PCA scatter projection and a side-by-side comparative profiling tool for any two states.
3. **🧬 Cluster Profiles:** Standardized feature heatmaps ($z$-scores) with human-readable traffic-light color guides and multi-topic centroid radar fingerprints.
4. **💼 Strategic Action Matrix:** Policy and business simulator with pre-configured mission presets (*Youth Skilling, AgTech, Civic Engagement*) and CSV export.
5. **🤖 AI Executive Briefing:** Synthesized executive prompts ready to copy, with one-click direct launcher buttons for **ChatGPT**, **Google Gemini**, and **Claude**.
6. **🗺️ Regional Map:** High-resolution administrative map of India with isolated vector silhouette mode, cluster donut breakdown (Fig. 1), and interactive state dossiers.
7. **🩺 Model Quality Diagnostics:** Statistical health check translating Silhouette scores and Davies–Bouldin indices into plain-English confidence tiers.
8. **📖 About & Limitations:** Essential guidelines on Google Trends relative normalization, exploratory clustering limits, and responsible governance.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or newer
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/regional-search-interest-explorer.git
   cd regional-search-interest-explorer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open at `http://localhost:8501`.*

---

## 🔬 Methodology & Algorithmic Design

```
   Raw Google Trends Data
            │
            ▼
   Robust Preprocessing & Cleaning (Handles '<1', timestamps, metadata)
            │
            ▼
   StandardScaler Normalization (Z-score feature space: μ=0, σ=1)
            │
            ├─────────────────────────────────────────┐
            ▼                                         ▼
   K-Medians (Manhattan Distance)            K-Means (Euclidean Distance)
   - Minimizes L1 Absolute Error             - Minimizes Within-Cluster Sum of Squares (WCSS)
   - Resilient against search spikes         - Standard spherical centroids
            │                                         │
            └────────────────────┬────────────────────┘
                                 │
                                 ▼
                     Cluster Validation & PCA
                     - Silhouette Score (Separation)
                     - Davies-Bouldin Index (Compactness)
                     - 2D PCA Projection (Explained Variance)
                                 │
                                 ▼
                     Geospatial & Strategic Translation
                     - India Infographic Choropleth
                     - Tiered Action Matrix (Priority Scoring)
                     - AI Executive Synthesis Prompt
```

* **Standardization:** All search features are standardized ($z$-scores) so high-volume baseline topics do not overpower niche emerging search topics.
* **K-Medians Resilience:** By using the median along each dimension and Manhattan distance ($L_1$), the model resists artificial distortion from viral search anomalies.
* **Geospatial Integrity:** Bound against official RFC 7946 GeoJSON boundaries covering all 36 administrative states and union territories.

---

## 📂 Supported Data Formats

The app works seamlessly out of the box with the included 36-region India benchmark dataset, but also dynamically ingests:

1. **Direct Google Trends Regional CSVs:**
   - Downloads from Google Trends *"Interest by Subregion"* or *"Compared breakdown"*.
   - Automatically skips metadata rows (e.g. `Category: All categories`), parses dates, strips `%`, and normalizes minimal `<1` volume to `0.5`.
2. **Custom Multi-Topic CSVs:**
   - Any CSV containing a geographical column (`Region`, `State`, `City`, `Location`, etc.) and numeric search topic columns.
   - Automatically drops or recalculates metadata or pre-existing cluster columns.

---

## 📁 Repository Structure

```
regional_search_interest_explorer/
├── app.py                     # Main Streamlit application & interactive engine
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & presentation
└── data/
    ├── india_states.geojson   # High-resolution administrative boundary GeoJSON
    ├── regional_cluster_results.csv # 36-region India benchmark search dataset
    ├── batch_stability.csv    # Best-of-20 cluster stability benchmark metrics
    ├── cluster_profiles.csv   # Saved cluster centroids across active topics
    └── kmeans_results.csv     # Comparative Euclidean K-Means baseline data
```

---

## 🛡️ Responsible Data Use & Caveats

* **Relative, Not Absolute:** Google Trends measures search interest indexed between `0` and `100` relative to the peak region and timeframe—it does **not** reflect absolute query counts, census population need, or unemployment rates.
* **Exploratory Guidance:** Machine learning partitions are exploratory analytical lenses, not permanent demographic categories.
* **Actionable Pre-testing:** Always triangulate search-interest patterns with on-the-ground surveys, government statistical registries, and localized pilot tests before deploying capital or policy initiatives.

---

<div align="center">
  <sub>Developed for Advanced Unsupervised Machine Learning, Data Strategy & Public Policy Intelligence.</sub>
</div>
