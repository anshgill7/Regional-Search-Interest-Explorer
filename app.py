import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Regional Search Interest Explorer",
    page_icon="🌐",
    layout="wide",
)

DATA_PATH = os.path.join("data", "regional_cluster_results.csv")

st.title("🌐 Regional Search Interest Explorer")
st.caption(
    "Explore regional Google Trends profiles, compare clustering configurations, "
    "and identify search patterns that may warrant further investigation."
)

def rewind_geojson_for_plotly(geojson):
    """
    Ensure exterior polygon rings have clockwise winding (negative Cartesian signed area)
    as required by D3.geo / Plotly choropleths to prevent spherical polygon inversion.
    """
    def ring_area(ring):
        n = len(ring)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        return area / 2.0

    import copy
    new_geo = copy.deepcopy(geojson)
    for feature in new_geo.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            new_coords = []
            for idx, ring in enumerate(coords):
                area = ring_area(ring)
                if idx == 0 and area > 0:
                    new_coords.append(list(reversed(ring)))
                elif idx > 0 and area < 0:
                    new_coords.append(list(reversed(ring)))
                else:
                    new_coords.append(ring)
            geom["coordinates"] = new_coords

        elif gtype == "MultiPolygon":
            new_coords = []
            for poly in coords:
                new_poly = []
                for idx, ring in enumerate(poly):
                    area = ring_area(ring)
                    if idx == 0 and area > 0:
                        new_poly.append(list(reversed(ring)))
                    elif idx > 0 and area < 0:
                        new_poly.append(list(reversed(ring)))
                    else:
                        new_poly.append(ring)
                new_coords.append(new_poly)
            geom["coordinates"] = new_coords

    return new_geo

def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_and_clean_dataset(uploaded_file_or_path):
    """
    Load and clean a CSV file. Supports:
    1. Standard regional tables with 'Region' and topic columns.
    2. Raw Google Trends regional exports (with metadata header rows,
       date-stamped topic headers like 'Education: (1/1/25 - 1/1/26)',
       '<1' / '< 1' values, and percentage signs).
    """
    if uploaded_file_or_path is None:
        raise ValueError("No file provided.")

    if hasattr(uploaded_file_or_path, "read"):
        content_bytes = uploaded_file_or_path.read()
        uploaded_file_or_path.seek(0)
    elif isinstance(uploaded_file_or_path, bytes):
        content_bytes = uploaded_file_or_path
    else:
        with open(uploaded_file_or_path, "rb") as f:
            content_bytes = f.read()

    # Try common encodings
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = content_bytes.decode("utf-8", errors="replace")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The uploaded CSV file is empty.")

    # Find the header row (skipping Google Trends metadata lines like "Category: ...")
    geo_keywords = [
        "region", "subregion", "state", "country", "territory",
        "city", "location", "area", "geo", "dma", "metro", "province"
    ]
    header_idx = 0
    for i, line in enumerate(lines[:25]):
        tokens = [t.strip().strip('"').strip("'").lower() for t in line.split(",")]
        if tokens and any(tokens[0] == k or tokens[0].startswith(k) for k in geo_keywords):
            header_idx = i
            break
        if any(tok in geo_keywords for tok in tokens):
            header_idx = i
            break
        if len(tokens) >= 2 and i + 1 < len(lines):
            next_tokens = [nt.strip().strip('"') for nt in lines[i + 1].split(",")]
            num_numeric = sum(
                1 for nt in next_tokens[1:]
                if nt.replace("%", "").replace("<", "").strip().replace(".", "", 1).isdigit()
            )
            if num_numeric >= 1:
                header_idx = i
                break

    csv_data = "\n".join(lines[header_idx:])
    raw_df = pd.read_csv(pd.io.common.StringIO(csv_data))
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # Identify the region column
    region_col = None
    for col in raw_df.columns:
        col_lower = col.strip().lower()
        if any(k == col_lower or col_lower.startswith(k) for k in geo_keywords):
            region_col = col
            break

    if region_col is None:
        for col in raw_df.columns:
            s_clean = raw_df[col].astype(str).str.replace("%", "", regex=False).str.replace("<", "", regex=False).str.strip()
            if not pd.to_numeric(s_clean, errors="coerce").notnull().all():
                region_col = col
                break

    if region_col is None:
        region_col = raw_df.columns[0]

    # Clean topic column headers (e.g. "Education: (1/1/25 - 1/1/26)" -> "Education")
    rename_dict = {region_col: "Region"}
    for col in raw_df.columns:
        if col == region_col:
            continue
        cleaned = re.sub(r":\s*\(.*?\)$", "", str(col)).strip()
        cleaned = re.sub(r"\s*\([^)]*\)$", "", cleaned).strip()
        cleaned = cleaned.replace('"', '').replace("'", "").strip()
        if not cleaned:
            cleaned = str(col).strip()
        rename_dict[col] = cleaned

    df = raw_df.rename(columns=rename_dict)
    df["Region"] = df["Region"].astype(str).str.strip()
    df = df[df["Region"].str.len() > 0]
    df = df[~df["Region"].str.lower().isin(["region", "subregion", "state", "nan", "null", "none"])]

    # Potential topic columns: exclude non-feature and metadata columns
    excluded = {"region", "cluster", "pc1", "pc2"}
    candidate_topics = [
        c for c in df.columns
        if c.lower() not in excluded and not c.endswith("_z")
    ]

    valid_topics = []
    for col in candidate_topics:
        s = df[col].astype(str).str.strip()
        # Convert Google Trends '<1' (minimal non-zero search volume) to 0.5
        s = s.str.replace(r"^<\s*1%?$", "0.5", regex=True)
        s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
        numeric_series = pd.to_numeric(s, errors="coerce")
        if numeric_series.notnull().sum() >= max(1, len(df) * 0.3):
            df[col] = numeric_series
            valid_topics.append(col)

    if not valid_topics:
        raise ValueError("Could not find any numeric search-interest topic columns in the uploaded file.")

    # Fill missing values with 0.0 (Google Trends standard for negligible volume)
    df[valid_topics] = df[valid_topics].fillna(0.0)
    df = df.drop_duplicates(subset=["Region"]).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(f"At least three valid regions are required for clustering (found {len(df)}).")

    return df[["Region"] + valid_topics].copy(), valid_topics

def k_medians(X, k, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), k, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iter):
        distances = cdist(X, centroids, metric="cityblock")
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            np.median(X[labels == i], axis=0)
            if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = cdist(X, centroids, metric="cityblock")
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(distances[np.arange(len(labels)), labels]))
    return labels, centroids, inertia

def fit_kmedians_best(X, k, n_init=20):
    best = None
    for seed in range(n_init):
        labels, centers, inertia = k_medians(
            X, k, max_iter=100, random_state=seed
        )
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    return best

def safe_metrics(X, labels, distance):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return np.nan, np.nan
    sil = silhouette_score(X, labels, metric=distance)
    # sklearn's Davies-Bouldin score uses Euclidean geometry.
    dbi = davies_bouldin_score(X, labels)
    return float(sil), float(dbi)

# Sidebar: data source
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload a regional CSV", type=["csv"])

try:
    if uploaded is not None:
        data, detected_topics = load_and_clean_dataset(uploaded)
        data_source = f"Uploaded ({uploaded.name})"
    else:
        data, detected_topics = load_and_clean_dataset(DATA_PATH)
        data_source = "Project dataset"
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.sidebar.caption(f"Source: {data_source} · {len(data)} regions · {len(detected_topics)} detected topics")

# Topic selector: allows custom selection of active topics
selected_topics = st.sidebar.multiselect(
    "Active search topics",
    options=detected_topics,
    default=detected_topics,
    help="Select which search topics to include in the clustering model."
)

if len(selected_topics) < 1:
    st.warning("Please select at least one search topic from the sidebar.")
    st.stop()

TOPICS = selected_topics

# Sidebar: model controls
st.sidebar.header("🎯 Regional Grouping Settings")

alg_options = {
    "K-Medians (Recommended: Resilient to Outliers)": "K-Medians (Manhattan)",
    "K-Means (Standard Averages)": "K-Means (Euclidean)",
}
selected_alg_label = st.sidebar.selectbox(
    "Grouping Method",
    list(alg_options.keys()),
    help="How the model groups regions. K-Medians prevents isolated sudden search spikes from skewing the results; K-Means groups by standard arithmetic averages."
)
algorithm = alg_options[selected_alg_label]

max_k = min(8, len(data) - 1)
if max_k <= 2:
    k = 2
    st.sidebar.caption("Number of groups: 2 (maximum for dataset size)")
else:
    k = st.sidebar.slider(
        "Number of Regional Groups (K)",
        2, max_k, 2,
        help="Select how many distinct regional profiles or behavioral personas you want to discover."
    )

with st.sidebar.expander("⚙️ Advanced Model Tuning (Optional)", expanded=False):
    n_init = st.slider(
        "Stability Iterations",
        5, 30, 20, step=5,
        help="How many times the algorithm tests random starting points to find the most mathematically stable groups. 20 is the recommended default."
    )

run = st.sidebar.button("🚀 Update Grouping Model", type="primary", use_container_width=True)

# Keep computed result in session state; recalculate when controls or data change.
data_hash = int(pd.util.hash_pandas_object(data[["Region"] + TOPICS]).sum())
settings = (
    algorithm,
    k,
    n_init,
    tuple(data["Region"].tolist()),
    tuple(TOPICS),
    data_hash,
)
if "model_settings" not in st.session_state:
    st.session_state.model_settings = None

if run or st.session_state.model_settings != settings:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[TOPICS].to_numpy(dtype=float))

    if algorithm == "K-Means (Euclidean)":
        model = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = model.fit_predict(X_scaled)
        centers = model.cluster_centers_
        objective = float(model.inertia_)
        distance = "euclidean"
        objective_name = "WCSS"
    else:
        labels, centers, objective = fit_kmedians_best(
            X_scaled, k, n_init=n_init
        )
        distance = "manhattan"
        objective_name = "Manhattan inertia"

    silhouette, dbi = safe_metrics(X_scaled, labels, distance)

    # PCA is for visual exploration, not the clustering itself.
    n_comp = min(2, len(TOPICS), len(data))
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)

    result = data.copy()
    result["Cluster"] = labels.astype(int)
    result["PC1"] = coords[:, 0]
    result["PC2"] = coords[:, 1] if n_comp > 1 else 0.0
    for idx, topic in enumerate(TOPICS):
        result[f"{topic}_z"] = X_scaled[:, idx]

    profiles = pd.DataFrame(X_scaled, columns=TOPICS)
    profiles["Cluster"] = labels
    profiles = profiles.groupby("Cluster")[TOPICS].mean()

    st.session_state.model_settings = settings
    st.session_state.result = result
    st.session_state.profiles = profiles
    st.session_state.metrics = {
        "silhouette": silhouette,
        "dbi": dbi,
        "objective": objective,
        "objective_name": objective_name,
        "distance": distance,
        "variance": float(pca.explained_variance_ratio_.sum()),
    }
    st.session_state.scaled = X_scaled

result = st.session_state.result
profiles = st.session_state.profiles
metrics = st.session_state.metrics

# Header metrics
counts = result["Cluster"].value_counts().sort_index()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regions analysed", len(result))
c2.metric("Topics", len(TOPICS))
c3.metric("Clusters", result["Cluster"].nunique())
c4.metric("Largest cluster", int(counts.max()))

tabs = st.tabs([
    "Overview",
    "Regional explorer",
    "Cluster profiles",
    "Strategic action matrix",
    "🤖 AI Executive Briefing",
    "Regional map",
    "Model evaluation",
    "About & limitations",
])

with tabs[0]:
    st.subheader("Regional search-interest overview")
    left, right = st.columns([1.15, 1])
    with left:
        chart_data = counts.rename("Regions").reset_index()
        chart_data.columns = ["Cluster", "Regions"]
        chart_data["Cluster"] = (
            "Cluster " + chart_data["Cluster"].astype(str)
        )

        fig = px.bar(
            chart_data,
            x="Cluster",
            y="Regions",
            color="Cluster",
            title="Regional Distribution Across Clusters",
            text="Regions",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Cluster",
            yaxis_title="Number of Regions",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="cluster_distribution_chart"
        )
    with right:
        st.markdown("### 📋 Executive Summary")
        dom_c = int(counts.idxmax())
        dom_n = int(counts.max())
        pct = (dom_n / len(result)) * 100
        st.markdown(
            f"""
            - **Grouping Approach:** `{algorithm}`
            - **Discovered Personas:** **{k} distinct regional clusters**
            - **Dominant Group:** **Cluster {dom_c}** ({dom_n} regions, {pct:.0f}% of total)
            - **Active Search Topics ({len(TOPICS)}):** {', '.join(f'`{t}`' for t in TOPICS)}
            """
        )
        with st.expander("📐 Technical Model Parameters", expanded=False):
            st.caption("Mathematical metrics for data science and statistical validation:")
            st.write(f"- Distance Metric: {metrics['distance'].title()}")
            st.write(f"- Objective ({metrics['objective_name']}): {metrics['objective']:.2f}")
            st.write(f"- 2D Projection Variance: {metrics['variance']:.1%}")
            st.write(f"- Tested Iterations: {n_init}")

    st.subheader("Regional assignments")
    st.dataframe(
        result[["Region", "Cluster"] + TOPICS].sort_values(["Cluster", "Region"]),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Explore Regional Behavioral Profiles")
    st.caption(
        "💡 **How to read this chart:** Each dot represents a region. States plotted close together share similar relative search behaviors across your selected topics, while distant points represent contrasting regional profiles."
    )
    selected_cluster = st.selectbox(
        "Filter by cluster",
        ["All clusters"] + [f"Cluster {int(x)}" for x in sorted(result["Cluster"].unique())],
        key="region_cluster_filter",
    )
    filtered = result.copy()
    if selected_cluster != "All clusters":
        c_num = int(selected_cluster.replace("Cluster ", ""))
        filtered = filtered[filtered["Cluster"] == c_num]

    fig = px.scatter(
        filtered,
        x="PC1",
        y="PC2",
        color=filtered["Cluster"].astype(str),
        hover_name="Region",
        hover_data=TOPICS,
        title="2D Map of Regional Search Behaviors (PCA Projection)",
        labels={
            "PC1": "Primary Interest Pattern (Dimension 1)",
            "PC2": "Secondary Interest Pattern (Dimension 2)",
            "color": "Cluster"
        },
    )
    fig.update_layout(legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: This 2D projection simplifies multiple search topics into two primary dimensions for visual exploration."
    )

    st.subheader("Compare two regions")
    region_names = result["Region"].tolist()
    if len(region_names) >= 2:
        a, b = st.columns(2)
        region_a = a.selectbox("First region", region_names, index=0)
        region_b = b.selectbox(
            "Second region", region_names,
            index=1 if len(region_names) > 1 else 0
        )
        compare = result.set_index("Region")
        if region_a == region_b:
            st.info("Choose two different regions to compare their profiles.")
        else:
            ca, cb = compare.loc[region_a], compare.loc[region_b]
            st.write(
                f"**Cluster membership:** {region_a} → {int(ca['Cluster'])} · "
                f"{region_b} → {int(cb['Cluster'])}"
            )
            comp = pd.DataFrame({
                "Topic": TOPICS,
                region_a: [ca[t] for t in TOPICS],
                region_b: [cb[t] for t in TOPICS],
            }).melt(id_vars="Topic", var_name="Region", value_name="Search interest")
            fig = px.bar(
                comp, x="Topic", y="Search interest", color="Region",
                barmode="group", title="Topic-interest comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Cluster Topic Fingerprints")
    st.markdown(
        """
        <div style='background:#1e293b;padding:12px 16px;border-radius:8px;margin-bottom:14px;border-left:4px solid #38bdf8;'>
        <b>💡 How to Read Topic Scores (Relative to National Average):</b><br>
        <span style='color:#34d399;'>🟢 <b>Positive (+)</b></span>: Search interest is <b>higher</b> than national baseline &nbsp;|&nbsp; 
        <span style='color:#94a3b8;'>⚪ <b>Zero (0.0)</b></span>: Exactly at national average &nbsp;|&nbsp; 
        <span style='color:#f87171;'>🔴 <b>Negative (−)</b></span>: Search interest is <b>lower</b> than national baseline
        </div>
        """,
        unsafe_allow_html=True
    )
    fig = px.imshow(
        profiles,
        text_auto=".2f",
        aspect="auto",
        color_continuous_midpoint=0,
        labels=dict(x="Search Topic", y="Regional Cluster", color="Relative Index"),
        title="Topic Fingerprint Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar or Bar Comparison of Cluster Centroids
    st.subheader("Centroid Comparison Across Topics")
    if len(TOPICS) >= 3:
        st.caption("Visual fingerprint of each cluster centroid across standardized feature space.")
        fig_radar = go.Figure()
        radar_topics = TOPICS + [TOPICS[0]]
        palette = px.colors.qualitative.Set2
        for i, c_idx in enumerate(profiles.index):
            vals = profiles.loc[c_idx].tolist()
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_topics,
                fill="toself",
                name=f"Cluster {c_idx}",
                line=dict(color=palette[i % len(palette)], width=2),
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showline=True)),
            showlegend=True,
            height=450,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="cluster_centroid_radar")
    else:
        st.caption("Bar comparison of cluster centroids across active topics.")
        profiles_melted = profiles.reset_index().melt(
            id_vars="Cluster", var_name="Topic", value_name="Mean z-score"
        )
        profiles_melted["Cluster"] = "Cluster " + profiles_melted["Cluster"].astype(str)
        fig_bar_centroid = px.bar(
            profiles_melted,
            x="Topic",
            y="Mean z-score",
            color="Cluster",
            barmode="group",
            title="Cluster Centroids Comparison",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_bar_centroid, use_container_width=True, key="cluster_centroid_bar")

    selected = st.selectbox(
        "Inspect a cluster",
        [int(x) for x in profiles.index],
        key="profile_cluster",
    )
    p = profiles.loc[selected].sort_values(ascending=False)
    profile_table = p.rename("Mean standardised interest").reset_index()
    profile_table.columns = ["Topic", "Mean standardised interest"]
    st.write(f"**Cluster {selected}: {int(counts.loc[selected])} regions**")
    st.dataframe(profile_table, use_container_width=True, hide_index=True)

    members = result.loc[result["Cluster"] == selected, "Region"].sort_values().tolist()
    st.write("**Member regions**")
    st.write(", ".join(members))

with tabs[3]:
    st.subheader("Strategic Action Matrix & Policy Simulator")
    st.caption(
        "Bridge unsupervised clustering outputs with real-world decision-making. "
        "Select a strategic domain or customize weights to calculate priority tiers, "
        "explore actionable playbooks, and export regional intervention plans."
    )

    is_uploaded = uploaded is not None
    if is_uploaded:
        st.success(f"📊 **Active Dataset:** Uploaded `{uploaded.name}` · {len(result)} regions · {len(TOPICS)} active search topics")
    else:
        st.info(f"📊 **Active Dataset:** Benchmark Project Dataset · {len(result)} regions · {len(TOPICS)} search topics")

    st.info("💡 **Looking for AI analysis?** Check out the dedicated **🤖 AI Executive Briefing** tab above to copy an AI-ready prompt synthesized from your active scenario.")

    DEFAULT_SET = {"Education", "Jobs", "Agriculture", "Politics", "Sports", "Tourism"}
    has_default_topics = DEFAULT_SET.issubset(set(TOPICS))

    domain_presets = {
        "🎓 Youth Employment & Skilling (Public Policy & EdTech)": {
            "weights": {"Education": 0.5, "Jobs": 0.5, "Agriculture": 0.0, "Politics": 0.0, "Sports": -0.2, "Tourism": 0.0},
            "desc": "Prioritizes regions with strong digital demand for education, skilling, and job search opportunities.",
            "policy": [
                "Establish localized District Skilling Centers in top Tier 1 regions.",
                "Organize regional youth apprenticeship programs and career counseling fairs.",
                "Expand high-speed broadband in educational institutions and community learning hubs."
            ],
            "business": [
                "Deploy vernacular EdTech test-prep bootcamps with localized subscription pricing.",
                "Target performance marketing campaigns for blue/grey-collar job portals.",
                "Partner with regional colleges for campus-to-corporate upskilling certifications."
            ],
        },
        "🌾 Agrarian Support & Rural Credit (AgriTech & Farm Support)": {
            "weights": {"Education": 0.0, "Jobs": 0.0, "Agriculture": 1.0, "Politics": 0.0, "Sports": 0.0, "Tourism": 0.0},
            "desc": "Identifies agricultural regions where farming communities actively seek agronomic information and support.",
            "policy": [
                "Scale agricultural advisory clinics and mobile soil-testing vans in priority farming districts.",
                "Drive saturation drives for agricultural credit cards and crop insurance.",
                "Deploy automated vernacular SMS/WhatsApp weather and crop advisory channels."
            ],
            "business": [
                "AgriTech market entry: roll out digital input marketplaces and direct-to-farm supply chains.",
                "Rural FinTech: offer digital crop loans and parametric weather micro-insurance.",
                "Smart farming: introduce pay-per-use solar pumps and mechanized rental services."
            ],
        },
        "⚽ Sports Ecosystem & Youth Engagement (Sports Infra & Gaming)": {
            "weights": {"Education": -0.2, "Jobs": -0.2, "Agriculture": 0.0, "Politics": 0.0, "Sports": 1.0, "Tourism": 0.0},
            "desc": "Highlights regions where sports, athletics, and recreational enthusiasm significantly outpace other search topics.",
            "policy": [
                "Establish State Centers of Excellence and grassroots athletic academies in high-interest regions.",
                "Upgrade regional stadiums with synthetic tracks, football turfs, and multi-sport facilities.",
                "Introduce structured school sports leagues and sports quota recruitment drives."
            ],
            "business": [
                "Sports retail & apparel expansion: open regional outlets and athletic footwear stores.",
                "Digital gaming & fantasy leagues: focus user acquisition campaigns on receptive regional markets.",
                "Private sports coaching academies and sports physiotherapy clinics."
            ],
        },
        "🏖️ Tourism Corridors & Hospitality (Regional Travel & Expansion)": {
            "weights": {"Education": 0.0, "Jobs": 0.0, "Agriculture": 0.0, "Politics": 0.0, "Sports": 0.0, "Tourism": 1.0},
            "desc": "Focuses on tourism hotspots and destinations with elevated search interest in travel and leisure.",
            "policy": [
                "Incentivize local homestay registration and eco-tourism hospitality standards.",
                "Upgrade last-mile scenic road corridors, wayside amenities, and digital tourist passes.",
                "Launch seasonal state tourism circuit marketing in major outbound source hubs."
            ],
            "business": [
                "Hospitality expansion: launch boutique hotels, experiential glamping, and backpacker hostels.",
                "OTA aggregators: partner with local tour operators for curated vernacular itineraries.",
                "Car rentals, adventure sports, and travel insurance micro-packages."
            ],
        },
        "🏛️ Civic Engagement & Public Information (Governance & Awareness)": {
            "weights": {"Education": 0.0, "Jobs": 0.0, "Agriculture": 0.0, "Politics": 1.0, "Sports": 0.0, "Tourism": 0.0},
            "desc": "Maps regions with elevated public interest in political developments, policy, and civic affairs.",
            "policy": [
                "Intensify voter education campaigns and digital electoral literacy drives.",
                "Expand digital citizen service portals and grievance redressal town halls.",
                "Publish localized open-data dashboards detailing welfare scheme progress."
            ],
            "business": [
                "Civic-tech & public affairs consulting: local sentiment and policy tracking.",
                "Vernacular news platforms & regional content creator networks.",
                "Public relations and community engagement programs for CSR initiatives."
            ],
        },
    }

    # If the user uploaded a custom dataset, prioritize dynamic data-driven options
    if is_uploaded:
        domain_options = [
            "⚖️ Balanced Composite Index (Equal Weights across active topics)",
            "⚙️ Custom Weighted Mission Simulator",
        ] + [f"🎯 Single-Topic Priority: {t}" for t in TOPICS]
        if has_default_topics:
            domain_options += list(domain_presets.keys())
    else:
        if has_default_topics:
            domain_options = list(domain_presets.keys()) + [
                "⚖️ Balanced Composite Index (Equal Weights across active topics)",
                "⚙️ Custom Weighted Mission Simulator"
            ]
        else:
            domain_options = [
                "⚖️ Balanced Composite Index (Equal Weights across active topics)",
                "⚙️ Custom Weighted Mission Simulator",
            ] + [f"🎯 Single-Topic Priority: {t}" for t in TOPICS]

    selected_domain = st.selectbox(
        "Select Strategic Mission / Sector",
        domain_options,
        index=0,
        key="strategy_domain_select",
    )

    if selected_domain == "⚙️ Custom Weighted Mission Simulator":
        st.markdown("**Customize Weights for Policy / Strategy Simulation**")
        st.caption("Adjust topic weights between -1.0 and +1.0 to model custom strategic missions.")
        num_cols = min(3, max(1, len(TOPICS)))
        cols = st.columns(num_cols)
        active_weights = {}
        default_w = round(1.0 / max(1, len(TOPICS)), 2)
        for idx, t in enumerate(TOPICS):
            active_weights[t] = cols[idx % num_cols].slider(
                f"{t} weight", -1.0, 1.0, default_w, step=0.1, key=f"strat_w_{t}"
            )
        domain_desc = f"Custom user-weighted strategic mission simulation across {len(TOPICS)} active topics."
        policy_actions = [
            "Prioritize capital allocation in Tier 1 regions aligned with composite weight distribution.",
            "Establish cross-departmental task forces between matching ministries and regional agencies.",
            "Implement periodic KPI monitoring based on search-interest feedback loops."
        ]
        biz_actions = [
            "Tailor product value proposition and localized marketing to match high-indexing demand.",
            "Deploy regional pilot tests in top-ranking Tier 1 states before national rollout.",
            "Form localized distribution partnerships with regional market leaders."
        ]
    elif selected_domain == "⚖️ Balanced Composite Index (Equal Weights across active topics)":
        eq_w = round(1.0 / max(1, len(TOPICS)), 3)
        active_weights = {t: eq_w for t in TOPICS}
        domain_desc = f"Equal-weighted balanced composite across all {len(TOPICS)} active search topics ({', '.join(TOPICS)})."
        policy_actions = [
            "Promote comprehensive regional development addressing balanced search priorities across all topics.",
            "Allocate baseline infrastructure and digital support across multi-topic clusters.",
            "Monitor multi-sector progress through periodic regional public engagement reviews."
        ]
        biz_actions = [
            "Develop multi-category product offerings serving diverse regional demand profiles.",
            "Establish multi-channel customer acquisition strategies aligned with high-priority areas.",
            "Track cross-category adoption trends to guide dynamic resource allocation."
        ]
    elif selected_domain.startswith("🎯 Single-Topic Priority:"):
        focus_topic = selected_domain.replace("🎯 Single-Topic Priority:", "").strip()
        active_weights = {t: (1.0 if t == focus_topic else 0.0) for t in TOPICS}
        domain_desc = f"Targeted priority indexing focused exclusively on demand for {focus_topic}."
        policy_actions = [
            f"Concentrate strategic interventions, agency task forces, and budget allocations on {focus_topic}.",
            f"Launch specialized public awareness initiatives and service delivery channels for {focus_topic}.",
            f"Establish regional benchmarks and performance milestones centered on {focus_topic} adoption."
        ]
        biz_actions = [
            f"Focus sales, marketing, and channel expansion specifically around regional {focus_topic} demand.",
            f"Partner with prominent local stakeholders and organizations operating in the {focus_topic} space.",
            f"Optimize digital ad spend and campaign targeting for regions demonstrating peak {focus_topic} interest."
        ]
    else:
        preset = domain_presets[selected_domain]
        active_weights = {t: preset["weights"].get(t, 0.0) for t in TOPICS}
        domain_desc = preset["desc"]
        policy_actions = preset["policy"]
        biz_actions = preset["business"]

    # Compute composite score for all regions
    strategy_df = result.copy()
    weights_sum = sum(abs(active_weights.get(t, 0.0)) for t in TOPICS)
    norm_factor = weights_sum if weights_sum > 0 else 1.0

    scores = np.zeros(len(strategy_df))
    for t in TOPICS:
        scores += strategy_df[f"{t}_z"].to_numpy() * active_weights.get(t, 0.0)
    scores = scores / norm_factor

    strategy_df["Priority_Score"] = np.round(scores, 3)

    def assign_tier(s):
        if s >= 0.25:
            return "Tier 1: High Priority"
        elif s >= -0.15:
            return "Tier 2: Moderate Priority"
        else:
            return "Tier 3: Emerging / Baseline"

    strategy_df["Priority_Tier"] = strategy_df["Priority_Score"].apply(assign_tier)

    def find_driver(row):
        contributions = {t: row[f"{t}_z"] * active_weights.get(t, 0.0) for t in TOPICS}
        if any(v != 0 for v in contributions.values()):
            return max(contributions, key=contributions.get)
        return max(TOPICS, key=lambda t: row[f"{t}_z"])

    def find_secondary_driver(row, primary):
        other_topics = [t for t in TOPICS if t != primary]
        if not other_topics:
            return primary
        contributions = {t: row[f"{t}_z"] * active_weights.get(t, 0.0) for t in other_topics}
        if any(v != 0 for v in contributions.values()):
            return max(contributions, key=contributions.get)
        return max(other_topics, key=lambda t: row[f"{t}_z"])

    strategy_df["Key_Driver"] = strategy_df.apply(find_driver, axis=1)

    # Dynamic tailored action that directly reflects the uploaded dataset's values
    def generate_recommendation(row):
        tier = row["Priority_Tier"]
        driver = row["Key_Driver"]
        driver_z = float(row.get(f"{driver}_z", 0.0))
        raw_val = row.get(driver, None)
        second_t = find_secondary_driver(row, driver)
        c_num = int(row["Cluster"])

        raw_str = f"search index: {raw_val:.0f}, " if pd.notnull(raw_val) else ""

        if "Tier 1" in tier:
            return (
                f"🚨 High Priority (Cluster {c_num}): Peak demand in {driver} ({raw_str}{driver_z:+.1f}σ) "
                f"with secondary momentum in {second_t}. Deploy targeted initiatives, allocate capital, and expand local presence."
            )
        elif "Tier 2" in tier:
            return (
                f"📈 Growth Opportunity (Cluster {c_num}): Moderate interest in {driver} ({raw_str}{driver_z:+.1f}σ). "
                f"Expand digital outreach, partner with regional channels, and nurture demand in {second_t}."
            )
        else:
            return (
                f"🔍 Baseline / Monitor (Cluster {c_num}): Below-average relative demand in {driver} ({raw_str}{driver_z:+.1f}σ). "
                f"Maintain baseline coverage and observe search shifts before deploying major capital."
            )

    strategy_df["Tailored_Action"] = strategy_df.apply(generate_recommendation, axis=1)

    tier1_count = int((strategy_df["Priority_Tier"] == "Tier 1: High Priority").sum())
    top_region = strategy_df.sort_values(by="Priority_Score", ascending=False).iloc[0]
    dominant_cluster = strategy_df[strategy_df["Priority_Tier"] == "Tier 1: High Priority"]["Cluster"].mode()
    dom_cluster_str = f"Cluster {int(dominant_cluster.iloc[0])}" if not dominant_cluster.empty else "N/A"

    # Contextualize directives with actual regions from the uploaded dataset
    top_3_regions = strategy_df.sort_values(by="Priority_Score", ascending=False)["Region"].head(3).tolist()
    top_3_str = ", ".join(top_3_regions)

    if is_uploaded:
        policy_actions = [
            f"Prioritize public initiatives and capital allocation in top Tier 1 regions ({top_3_str}) indexing highest in {top_region['Key_Driver']}.",
            f"Deploy localized community programs and civic outreach addressing regional demand across {', '.join(TOPICS[:3])}.",
            f"Establish periodic trend feedback loops across regional clusters to gauge public adoption velocity."
        ]
        biz_actions = [
            f"Focus regional go-to-market teams, sales channels, and campaign spend in {top_region['Region']} ({top_region['Priority_Score']:+.2f} score) and {top_3_str}.",
            f"Tailor product value proposition and localized marketing to capitalize on peak demand in {top_region['Key_Driver']}.",
            f"Form regional distribution and channel partnerships to rapidly capture market share in high-priority {dom_cluster_str}."
        ]

    st.info(f"**Mission Scope:** {domain_desc}")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tier 1 (High Priority) Regions", f"{tier1_count} / {len(strategy_df)}")
    kpi2.metric("Top Priority Region", f"{top_region['Region']}")
    kpi3.metric("Highest Priority Score", f"{top_region['Priority_Score']:+.2f}")
    kpi4.metric("Leading Cluster", dom_cluster_str)

    s_col1, s_col2 = st.columns([1.6, 1])

    tier_colors = {
        "Tier 1: High Priority": "#E74C3C",
        "Tier 2: Moderate Priority": "#F39C12",
        "Tier 3: Emerging / Baseline": "#2ECC71",
    }

    with s_col1:
        st.markdown("**Ranked Regional Priority Index**")
        sorted_strat = strategy_df.sort_values(by="Priority_Score", ascending=True)
        fig_bar = px.bar(
            sorted_strat,
            x="Priority_Score",
            y="Region",
            orientation="h",
            color="Priority_Tier",
            color_discrete_map=tier_colors,
            title=f"Composite Priority Scores Across Regions ({data_source})",
            labels={"Priority_Score": "Priority Score", "Region": "Region"},
            height=max(400, min(800, len(strategy_df) * 22)),
            hover_data={t: True for t in TOPICS},
        )
        fig_bar.update_layout(
            legend_title_text="Priority Tier",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="strategic_priority_bar")

    with s_col2:
        st.markdown("**Priority Tier Breakdown by Cluster**")
        cluster_tier_matrix = pd.crosstab(
            strategy_df["Cluster"].astype(str),
            strategy_df["Priority_Tier"]
        ).reset_index()
        cluster_tier_matrix["Cluster"] = "Cluster " + cluster_tier_matrix["Cluster"]
        fig_matrix = px.bar(
            cluster_tier_matrix.melt(id_vars="Cluster", var_name="Priority Tier", value_name="Count"),
            x="Cluster",
            y="Count",
            color="Priority Tier",
            color_discrete_map=tier_colors,
            barmode="stack",
            title="Cluster Composition by Tier",
            height=320,
        )
        fig_matrix.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_matrix, use_container_width=True, key="cluster_tier_breakdown")

        st.markdown("**Active Topic Weights**")
        weight_df = pd.DataFrame({
            "Topic": TOPICS,
            "Assigned Weight": [active_weights.get(t, 0.0) for t in TOPICS]
        })
        st.dataframe(weight_df, hide_index=True, use_container_width=True)

    st.subheader("Actionable Strategic Playbook")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### 🏛️ Public Policy & Governance Directives")
        for act in policy_actions:
            st.markdown(f"- {act}")
    with p_col2:
        st.markdown("#### 💼 Commercial & Go-To-Market Playbook")
        for act in biz_actions:
            st.markdown(f"- {act}")

    st.subheader("Regional Strategic Action Plan")
    filter_tier = st.selectbox(
        "Filter Action Plan by Tier",
        ["All Tiers", "Tier 1: High Priority", "Tier 2: Moderate Priority", "Tier 3: Emerging / Baseline"],
        key="filter_action_plan_tier",
    )
    display_plan = strategy_df.copy()
    if filter_tier != "All Tiers":
        display_plan = display_plan[display_plan["Priority_Tier"] == filter_tier]

    plan_table = display_plan[[
        "Region", "Cluster", "Priority_Score", "Priority_Tier", "Key_Driver", "Tailored_Action"
    ]].sort_values(by="Priority_Score", ascending=False)

    st.dataframe(plan_table, hide_index=True, use_container_width=True)

    export_csv = plan_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Strategic Action Plan (CSV)",
        data=export_csv,
        file_name="strategic_regional_action_plan.csv",
        mime="text/csv",
        help="Download the prioritized intervention plan for reporting and presentation.",
        key="download_action_plan_csv",
    )

with tabs[4]:
    st.subheader("🤖 AI Executive Interpretation Assistant")
    st.caption(
        "Synthesize your active clustering results and priority index into an executive-ready analytical prompt. "
        "Click the copy button in the top-right corner of the prompt box and paste it directly into ChatGPT, Gemini, or Claude."
    )

    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Active Dataset", f"{len(result)} regions")
    ac2.metric("Topics Modeled", f"{len(TOPICS)} topics")
    ac3.metric("Top Priority Region", f"{top_region['Region']}")
    ac4.metric("Leading Cluster", dom_cluster_str)

    # Format cluster profile centroids
    centroid_summary_lines = []
    for c_idx in profiles.index:
        top_c_topics = profiles.loc[c_idx].sort_values(ascending=False)
        top_strengths = [f"{t} ({top_c_topics[t]:+.2f}σ)" for t in top_c_topics.index[:3]]
        members_c = result.loc[result["Cluster"] == c_idx, "Region"].tolist()
        centroid_summary_lines.append(
            f"- **Cluster {c_idx}** ({len(members_c)} regions): Top traits: {', '.join(top_strengths)}. Sample regions: {', '.join(members_c[:4])}."
        )
    centroid_summary_str = "\n".join(centroid_summary_lines)

    tier1_regions_list = strategy_df[strategy_df["Priority_Tier"] == "Tier 1: High Priority"].sort_values(by="Priority_Score", ascending=False)
    tier1_items = [
        f"  * {r['Region']} (Score: {r['Priority_Score']:+.2f}, Top Driver: {r['Key_Driver']})"
        for _, r in tier1_regions_list.head(8).iterrows()
    ]
    tier1_summary_str = "\n".join(tier1_items) if tier1_items else "  * No regions currently categorized in Tier 1."

    ai_prompt_text = f"""Act as a Senior Data & Public Policy Strategist. Analyze the following unsupervised machine learning clustering results from a regional search-interest study.

### 1. Analysis Context & Model Parameters
- Dataset: {data_source} ({len(result)} regions, {len(TOPICS)} search topics)
- Active Topics: {', '.join(TOPICS)}
- Algorithm: {algorithm} (K={k}, distance: {metrics['distance']})
- Quality Metrics: Silhouette Score = {metrics['silhouette']:.4f}, Davies-Bouldin Index = {metrics['dbi']:.4f}, PCA Explained Variance (2D) = {metrics['variance']:.1%}
- Active Mission Focus: {domain_desc}

### 2. Cluster Fingerprints (Centroids in Standardized Space)
{centroid_summary_str}

### 3. Priority Index & Tier 1 Leading Regions
- Highest Priority Score: {top_region['Region']} ({top_region['Priority_Score']:+.2f})
- Dominant Cluster in Tier 1: {dom_cluster_str}
- Top Priority Regions:
{tier1_summary_str}

### 4. Required Analysis & Deliverables
Please synthesize these results into an executive briefing covering:
1. Behavioral Fingerprints: What distinct public interest profiles or behavioral personas do these clusters represent?
2. Targeted Interventions: For the Tier 1 priority regions, what specific public policy initiatives or localized programs would you recommend?
3. Commercial & Go-To-Market Playbook: How can businesses and digital services tailor their offerings to these distinct regional demand profiles?
4. Responsible Use & Data Caveats: Given that Google Trends measures relative search interest (not absolute volume, public need, or unemployment rates), what critical limitations must leadership keep in mind?
"""

    st.markdown("### 📋 Ready-to-Copy Engineered Prompt")
    st.caption("Click the copy icon in the top right of the code block:")
    st.code(ai_prompt_text, language="markdown")

    st.markdown("### 🚀 Paste Directly Into Your Preferred AI Platform")
    col_link1, col_link2, col_link3 = st.columns(3)
    col_link1.link_button("👉 Open ChatGPT", "https://chatgpt.com", use_container_width=True)
    col_link2.link_button("👉 Open Google Gemini", "https://gemini.google.com", use_container_width=True)
    col_link3.link_button("👉 Open Claude", "https://claude.ai", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Follow-Up Prompts to Ask the AI:")
    st.markdown(
        """
        - *\"What specific risks might emerge if policy decisions solely target Tier 1 regions and ignore Tier 3 baseline regions?\"*
        - *\"How can we design an A/B pilot testing framework in the top-ranking region before broad expansion?\"*
        - *\"Summarize the key differences between the cluster centroids in a 1-page memo for executive leadership.\"*
        """
    )

# REGIONAL MAP
with tabs[5]:
    st.subheader("🇮🇳 India Regional Cluster Infographic Map")
    st.caption(
        "Interactive geospatial infographic displaying regional search-interest clusters with crisp state boundaries, "
        "cluster distribution metrics, and detailed topic breakdown profiles."
    )
    geojson_path = os.path.join("data", "india_states.geojson")

    if not os.path.exists(geojson_path):
        st.error("India GeoJSON file not found.")
    else:
        with open(geojson_path, "r", encoding="utf-8") as f:
            india_geojson = json.load(f)

        # Prepare data for mapping
        map_data = result.copy()
        map_data["Region"] = map_data["Region"].astype(str).str.strip()
        # Merge small union territories and align state naming conventions
        combined_ut = "Dadra and Nagar Haveli and Daman and Diu"
        map_data["Map_Region"] = map_data["Region"].replace({
            "Dadra and Nagar Haveli": combined_ut,
            "Daman and Diu": combined_ut,
            "Andaman and Nicobar Islands": "Andaman & Nicobar",
            "Jammu and Kashmir": "Jammu & Kashmir",
        })

        # Aggregate metrics per region
        def cluster_assignment(series):
            vals = series.unique()
            return int(vals[0]) if len(vals) == 1 else -1

        aggregation = {
            "Cluster": ("Cluster", cluster_assignment),
            "Region": ("Region", lambda x: ", ".join(x)),
        }
        for topic in TOPICS:
            aggregation[topic] = (topic, "mean")

        map_plot_data = (
            map_data
            .groupby("Map_Region")
            .agg(**aggregation)
            .reset_index()
        )

        # Human-readable cluster label
        map_plot_data["Cluster_Label"] = map_plot_data["Cluster"].apply(
            lambda c: "Mixed / Merged Regions" if c == -1 else f"Cluster {int(c)}"
        )

        # Include Ladakh explicitly with "No Data Available" status
        if "Ladakh" not in map_plot_data["Map_Region"].values:
            ladakh_row = {
                "Map_Region": "Ladakh",
                "Region": "Ladakh (No separate Google Trends data)",
                "Cluster": -99,
                "Cluster_Label": "No Data Available",
            }
            for topic in TOPICS:
                ladakh_row[topic] = np.nan
            map_plot_data = pd.concat([map_plot_data, pd.DataFrame([ladakh_row])], ignore_index=True)

        # Verify mapping to GeoJSON features
        geo_names = {f["properties"]["name"] for f in india_geojson["features"]}
        matched = map_plot_data[map_plot_data["Map_Region"].isin(geo_names)].copy()
        unmatched = sorted(set(map_plot_data["Map_Region"]) - geo_names)
        missing_data = sorted(geo_names - set(map_plot_data["Map_Region"]))

        # KPI Header row
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mapped Regions", f"{len(matched)} / {len(geo_names)}")
        active_clusters = matched[matched["Cluster"] >= 0]["Cluster"].nunique()
        mc2.metric("Clusters Formed", f"{active_clusters}")
        valid_clusters = matched[matched["Cluster_Label"] != "No Data Available"]["Cluster_Label"]
        largest_c = valid_clusters.value_counts().idxmax() if not valid_clusters.empty else "N/A"
        largest_n = valid_clusters.value_counts().max() if not valid_clusters.empty else 0
        mc3.metric("Dominant Cluster", f"{largest_c} ({largest_n} states)")
        mc4.metric("Unmatched Regions", len(unmatched))

        if len(matched) > 0:
            if unmatched:
                st.warning(f"Dataset regions not found in GeoJSON: {', '.join(unmatched)}")
            if missing_data:
                st.caption(f"Features in GeoJSON without search data: {', '.join(missing_data)}")

            matched = matched.sort_values(by="Cluster")

            # Dynamic qualitative color mapping for crisp, attractive cluster contrast
            unique_clusters = sorted(matched["Cluster_Label"].unique())
            qualitative_palette = [
                "#38bdf8",  # Sky Blue / Cyan (matches infographic style)
                "#f59e0b",  # Amber / Warm Gold
                "#34d399",  # Emerald Green
                "#a78bfa",  # Soft Violet
                "#f87171",  # Soft Coral
                "#fb923c",  # Warm Orange
                "#ec4899",  # Rose Pink
                "#2dd4bf",  # Teal
            ]
            color_discrete_map = {}
            palette_idx = 0
            for cl in unique_clusters:
                if "no data" in cl.lower():
                    color_discrete_map[cl] = "#64748b"  # Muted Slate Neutral for No Data
                elif "mixed" in cl.lower():
                    color_discrete_map[cl] = "#94a3b8"  # Slate Neutral for Merged
                else:
                    color_discrete_map[cl] = qualitative_palette[palette_idx % len(qualitative_palette)]
                    palette_idx += 1

            # Rich, elegant HTML tooltip on hover showing state and topic values
            def build_hover(row):
                if row["Cluster_Label"] == "No Data Available":
                    lines = [
                        f"<b style='font-size:15px;color:#ffffff;'>📍 {row['Map_Region']}</b>",
                        "<span style='color:#94a3b8;'>Status:</span> <b style='color:#f87171;'>No Data Available</b>",
                        "<span style='color:#475569;'>───────────────────────────</span>",
                        "<span style='color:#cbd5e1;font-size:12px;'>Google Trends does not report search interest</span>",
                        "<span style='color:#cbd5e1;font-size:12px;'>for Ladakh separately (bundled with J&K).</span>",
                    ]
                    return "<br>".join(lines)

                lines = [
                    f"<b style='font-size:15px;color:#ffffff;'>📍 {row['Map_Region']}</b>",
                    f"<span style='color:#94a3b8;'>Cluster:</span> <b style='color:#38bdf8;'>{row['Cluster_Label']}</b>",
                ]
                if row['Region'] != row['Map_Region']:
                    lines.append(f"<span style='color:#64748b;font-size:11px;'>Constituents: {row['Region']}</span>")
                lines.append("<span style='color:#475569;'>───────────────────────────</span>")
                lines.append("<b style='color:#e2e8f0;'>Search Interest Profile:</b>")
                for t in TOPICS:
                    val = row[t]
                    val_str = f"{val:.1f}" if pd.notnull(val) else "N/A"
                    lines.append(f"• <span style='color:#cbd5e1;'>{t}:</span> <b style='color:#ffffff;'>{val_str}</b>")
                return "<br>".join(lines)

            matched["hover_info"] = matched.apply(build_hover, axis=1)

            # Two-column Infographic Dashboard Layout (matching user reference image)
            map_col, stats_col = st.columns([1.55, 1], gap="medium")

            with map_col:
                map_style_choice = st.radio(
                    "Display Style",
                    ["🎨 Clean Infographic Vector (Isolated India)", "🌐 Tile Basemap View"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="map_display_style"
                )

                if map_style_choice == "🎨 Clean Infographic Vector (Isolated India)":
                    fig_map = px.choropleth(
                        matched,
                        geojson=india_geojson,
                        locations="Map_Region",
                        featureidkey="properties.name",
                        color="Cluster_Label",
                        color_discrete_map=color_discrete_map,
                        custom_data=["hover_info"],
                    )
                    fig_map.update_geos(
                        fitbounds="locations",
                        visible=False,
                    )
                    fig_map.update_traces(
                        hovertemplate="%{customdata[0]}<extra></extra>",
                        marker_line_width=1.5,
                        marker_line_color="#0f172a",
                    )
                    fig_map.update_layout(
                        height=720,
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        hoverlabel=dict(
                            bgcolor="#0f172a",
                            font_size=13,
                            font_family="Inter, Roboto, sans-serif",
                            font_color="#f8fafc",
                            bordercolor="#38bdf8",
                        ),
                        legend_title_text="Cluster",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    # Tile basemap view (cross-version compatible)
                    fig_map = None
                    if hasattr(px, "choropleth_map"):
                        try:
                            fig_map = px.choropleth_map(
                                matched,
                                geojson=india_geojson,
                                locations="Map_Region",
                                featureidkey="properties.name",
                                color="Cluster_Label",
                                color_discrete_map=color_discrete_map,
                                custom_data=["hover_info"],
                                title="Regional Search-Interest Clusters (India)",
                                map_style="carto-positron",
                                zoom=3.8,
                                center={"lat": 22.5, "lon": 82.0},
                                opacity=0.85,
                            )
                        except Exception:
                            fig_map = None

                    if fig_map is None and hasattr(px, "choropleth_mapbox"):
                        try:
                            fig_map = px.choropleth_mapbox(
                                matched,
                                geojson=india_geojson,
                                locations="Map_Region",
                                featureidkey="properties.name",
                                color="Cluster_Label",
                                color_discrete_map=color_discrete_map,
                                custom_data=["hover_info"],
                                title="Regional Search-Interest Clusters (India)",
                                mapbox_style="carto-positron",
                                zoom=3.8,
                                center={"lat": 22.5, "lon": 82.0},
                                opacity=0.85,
                            )
                        except Exception:
                            fig_map = None

                    if fig_map is None:
                        fig_map = px.choropleth(
                            matched,
                            geojson=india_geojson,
                            locations="Map_Region",
                            featureidkey="properties.name",
                            color="Cluster_Label",
                            color_discrete_map=color_discrete_map,
                            custom_data=["hover_info"],
                        )
                        fig_map.update_geos(fitbounds="locations", visible=False)

                    fig_map.update_traces(
                        hovertemplate="%{customdata[0]}<extra></extra>",
                        marker_line_width=1.2,
                        marker_line_color="#1e293b",
                    )
                    fig_map.update_layout(
                        height=720,
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        hoverlabel=dict(
                            bgcolor="#0f172a",
                            font_size=13,
                            font_family="Inter, Roboto, sans-serif",
                            font_color="#f8fafc",
                            bordercolor="#38bdf8",
                        ),
                        legend_title_text="Cluster",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

            with stats_col:
                st.markdown("#### 🍩 Cluster Breakdown (FIG. 1)")
                c_counts = matched["Cluster_Label"].value_counts().reset_index()
                c_counts.columns = ["Cluster", "Regions"]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=c_counts["Cluster"],
                    values=c_counts["Regions"],
                    hole=0.62,
                    marker=dict(
                        colors=[color_discrete_map.get(c, "#38bdf8") for c in c_counts["Cluster"]],
                        line=dict(color="#0f172a", width=2)
                    ),
                    textinfo="label+percent",
                    textposition="outside",
                    showlegend=False,
                )])
                fig_donut.update_layout(
                    height=230,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", size=11),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

                st.markdown("#### 📊 Mean Search Interest by Cluster (FIG. 2)")
                c_means = matched[matched["Cluster_Label"] != "No Data Available"].groupby("Cluster_Label")[TOPICS].mean().reset_index()
                c_melt = c_means.melt(id_vars="Cluster_Label", var_name="Topic", value_name="Interest")
                fig_bar_topics = px.bar(
                    c_melt,
                    x="Interest",
                    y="Topic",
                    color="Cluster_Label",
                    color_discrete_map=color_discrete_map,
                    barmode="group",
                    orientation="h",
                    height=220,
                )
                fig_bar_topics.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis_title="Avg Interest",
                    yaxis_title="",
                )
                st.plotly_chart(fig_bar_topics, use_container_width=True)

                st.markdown("#### 📍 Regional Dossier")
                sorted_state_names = sorted(matched["Map_Region"].unique())
                inspect_state = st.selectbox(
                    "Inspect state data",
                    sorted_state_names,
                    key="map_inspect_state_select",
                    label_visibility="collapsed"
                )
                state_row = matched[matched["Map_Region"] == inspect_state].iloc[0]
                badge_color = color_discrete_map.get(state_row["Cluster_Label"], "#38bdf8")
                st.markdown(
                    f"<div style='background:#1e293b;padding:10px 14px;border-radius:8px;border-left:4px solid {badge_color};margin-bottom:10px;'>"
                    f"<b style='font-size:15px;color:#f8fafc;'>{inspect_state}</b><br>"
                    f"<span style='color:#94a3b8;font-size:12px;'>Cluster Assignment: </span>"
                    f"<b style='color:{badge_color};font-size:13px;'>{state_row['Cluster_Label']}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if state_row["Cluster_Label"] == "No Data Available":
                    st.info("No separate Google Trends search interest data is available for Ladakh (historically bundled with Jammu & Kashmir).")
                else:
                    for t in TOPICS:
                        score = float(state_row[t])
                        st.caption(f"{t}: **{score:.1f}**")
                        st.progress(min(1.0, max(0.0, score / 100.0)))
        else:
            st.info(
                "None of the regions in this dataset match the built-in India state boundaries GeoJSON. "
                "Map choropleth visualization is available when region names match Indian states and union territories. "
                "Your clustering model, topic profiles, and strategic priority matrix are fully computed and viewable in the other tabs."
            )
            fig_bar = px.bar(
                result.sort_values(by=["Cluster", "Region"]),
                x="Region",
                y="Cluster",
                color=result["Cluster"].astype(str),
                title="Regional Cluster Assignments",
                labels={"Cluster": "Cluster ID"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tabs[6]:
    st.subheader("Model Quality & Grouping Diagnostics")
    st.caption("These statistical benchmarks evaluate how distinct, compact, and reliable the discovered regional clusters are.")

    sil = metrics["silhouette"]
    if sil > 0.35:
        health_status = "🟢 Strong Separation"
        health_desc = "The discovered clusters represent distinctly separated, well-defined regional profiles."
    elif sil > 0.15:
        health_status = "🟡 Balanced / Natural Overlap"
        health_desc = "The clusters capture clear directional behavioral tendencies with expected gradual transitions between neighboring states."
    else:
        health_status = "🟠 Subtle Distinction"
        health_desc = "Regional search interest is relatively uniform across the selected topics."

    st.markdown(
        f"<div style='background:#1e293b;padding:12px 18px;border-radius:10px;border-left:5px solid #38bdf8;margin-bottom:18px;'>"
        f"<b style='font-size:15px;color:#f8fafc;'>Diagnostic Assessment: {health_status}</b><br>"
        f"<span style='color:#cbd5e1;font-size:13px;'>{health_desc}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Cluster Separation",
        f"{metrics['silhouette']:.2f}",
        help="Silhouette Score: Measures how distinct clusters are from each other (-1 to +1). Higher values mean clearer separation."
    )
    m2.metric(
        "Cluster Cohesion",
        f"{metrics['dbi']:.2f}",
        help="Davies-Bouldin Index: Measures how tightly grouped states are within their cluster. Lower values indicate tighter, more coherent groups."
    )
    m3.metric(
        "2D Variance Captured",
        f"{metrics['variance']:.0%}",
        help="Percentage of the overall multi-topic behavioral variation captured in the 2D visual charts."
    )

    stability_path = os.path.join("data", "batch_stability.csv")
    if os.path.exists(stability_path):
        stability = pd.read_csv(stability_path)
        st.subheader("Saved best-of-20 batch stability")
        st.dataframe(stability.round(4), use_container_width=True, hide_index=True)
        fig = px.line(
            stability, x="K", y="Mean ARI", markers=True,
            title="Mean agreement across independent best-of-20 batches"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "These are previously computed project results from the 36-region India benchmark dataset, "
            "not a stability estimate for newly uploaded datasets."
        )

    st.subheader("Download current analysis")
    drop_cols = [c for c in ["PC1", "PC2"] + [f"{t}_z" for t in TOPICS] if c in result.columns]
    st.download_button(
        "Download regional assignments (CSV)",
        result.drop(columns=drop_cols).to_csv(index=False).encode("utf-8"),
        file_name="regional_cluster_assignments.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cluster profiles (CSV)",
        profiles.to_csv().encode("utf-8"),
        file_name="cluster_profiles.csv",
        mime="text/csv",
    )

with tabs[7]:
    st.subheader("How to interpret this application")
    st.markdown(
        f"""
        This application groups regions according to their relative search-interest
        profiles across **{', '.join(TOPICS)}**.

        **It identifies patterns in Google Trends data; it does not measure actual
        search counts, public need, unemployment, awareness, or service demand.**
        """
    )
    st.markdown(
        """
        **Important limitations**
        - Google Trends values are relative and depend on the selected comparison,
          geography and time period.
        - Low search interest does not prove low awareness or unmet need.
        - Topic selection and the selected time period influence the clusters.
        - K-Means and K-Medians are exploratory methods; cluster labels are not
          permanent categories of regions.
        - PCA is used for visualisation, not as proof of cluster validity.
        - The Davies–Bouldin index uses Euclidean geometry in this implementation.
        """
    )
    st.markdown(
        """
        **Responsible use:** Treat unusual or low search-interest patterns as
        prompts for further investigation. Combine them with comparable official
        statistics, surveys and local evidence before planning interventions.
        """
    )
