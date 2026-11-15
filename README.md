# Regional Search Interest Explorer

An interactive Streamlit application for exploring regional Google Trends search-interest profiles.

## Included data
- `regional_cluster_results.csv`: 36 regions, six topic features, and the exported K=2 cluster assignment.
- `batch_stability.csv`: saved best-of-20 batch stability results for K=2 through K=8.
- Other exported CSVs: cluster profiles, K-Means metrics, K-Medians metrics, and initialisation stability.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Supported upload formats

The application dynamically detects search topics and supports:

1. **Direct Google Trends exports**:
   - CSVs downloaded directly from Google Trends ("Interest by Subregion" or "Compared breakdown").
   - Automatically handles top metadata header lines (e.g. `Category: All categories`), date-stamped column names (e.g. `Education: (1/1/25 - 1/1/26)`), `<1` values (interpreted as minimal non-zero volume), and percentage symbols.

2. **Custom regional CSVs**:
   - Any CSV containing a region column (named `Region`, `Subregion`, `State`, `Country`, `City`, etc.) and any number of numeric search topic columns.
   - Any prior `Cluster`, `PC1`, `PC2`, or `*_z` columns are ignored and recalculated automatically.
   - An interactive topic selector in the sidebar allows you to select all or any subset of detected topics for analysis.

## Methodology notes

- Input topic features are standardised before clustering.
- K-Means uses Euclidean distance.
- K-Medians uses Manhattan distance and median centroids, selecting the lowest Manhattan-inertia solution from the configured number of initialisations.
- Silhouette uses the algorithm's corresponding distance.
- Scikit-learn's Davies–Bouldin index uses Euclidean geometry, including when evaluating K-Medians labels.
- PCA is a two-dimensional visualisation, not the clustering space.
- Google Trends interest is relative search interest, not actual search counts or a direct measure of public need.

## Presentation note

The exported project labels are the saved K-Medians K=2 assignments. The dashboard recomputes clustering from the six topic columns when opened, so the cluster numbers may be relabelled while the partition remains equivalent.
