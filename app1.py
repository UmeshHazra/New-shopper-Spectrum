"""
Shopper Spectrum — Customer Segmentation & Product Recommendation Dashboard
============================================================================
Run with:  streamlit run app.py

Expects an "Online Retail" style transaction CSV with columns:
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

You can either:
  1. Place your CSV at ./data/online_retail.csv, or
  2. Upload a CSV from the sidebar at runtime.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="ISO-8859-1")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Drop missing CustomerID / Description
    df = df.dropna(subset=["CustomerID", "Description"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid quantities / prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


@st.cache_data(show_spinner="Building demo dataset...")
def make_demo_data(n_customers=300, n_products=40, seed=42) -> pd.DataFrame:
    """Synthetic fallback dataset so the dashboard runs without a file."""
    rng = np.random.default_rng(seed)
    products = [f"PRODUCT {i:03d}" for i in range(n_products)]
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain",
                 "Netherlands", "Australia", "Belgium", "Portugal", "Italy"]

    rows = []
    invoice_counter = 536000
    base_date = pd.Timestamp("2024-01-01")

    for cust in range(1, n_customers + 1):
        n_orders = rng.integers(1, 12)
        country = rng.choice(countries, p=[0.45, 0.1, 0.1, 0.05, 0.05,
                                            0.05, 0.05, 0.05, 0.05, 0.05])
        for _ in range(n_orders):
            invoice_counter += 1
            order_date = base_date + timedelta(days=int(rng.integers(0, 365)))
            n_items = rng.integers(1, 6)
            chosen = rng.choice(products, size=n_items, replace=False)
            for prod in chosen:
                rows.append({
                    "InvoiceNo": str(invoice_counter),
                    "StockCode": prod.split()[1],
                    "Description": prod,
                    "Quantity": int(rng.integers(1, 10)),
                    "InvoiceDate": order_date,
                    "UnitPrice": round(float(rng.uniform(2, 80)), 2),
                    "CustomerID": str(1000 + cust),
                    "Country": country,
                })
    df = pd.DataFrame(rows)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


# --------------------------------------------------------------------------
# RFM + Segmentation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculating RFM & segments...")
def compute_rfm(df: pd.DataFrame, n_clusters: int = 4):
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    # Rank clusters by average monetary value to assign human labels
    cluster_rank = (
        rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index
    )
    label_map = {}
    names = ["High-Value", "Regular", "Occasional", "At-Risk"]
    for i, c in enumerate(cluster_rank):
        label_map[c] = names[i] if i < len(names) else f"Segment {i}"
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    return rfm, scaler, km, label_map


def classify_new_customer(recency, frequency, monetary, rfm_reference, scaler, km, label_map):
    X = scaler.transform([[recency, frequency, monetary]])
    cluster = km.predict(X)[0]
    return label_map.get(cluster, f"Segment {cluster}")


# --------------------------------------------------------------------------
# Product recommendation (item-based collaborative filtering)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Building recommendation model...")
def build_similarity_matrix(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="CustomerID", columns="Description", values="Quantity", aggfunc="sum", fill_value=0
    )
    sim = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(sim, index=pivot.columns, columns=pivot.columns)
    return sim_df


def recommend_products(product_name: str, sim_df: pd.DataFrame, top_n: int = 5):
    if product_name not in sim_df.columns:
        # fuzzy fallback: case-insensitive partial match
        matches = [c for c in sim_df.columns if product_name.lower() in c.lower()]
        if not matches:
            return None, []
        product_name = matches[0]
    scores = sim_df[product_name].sort_values(ascending=False)
    scores = scores.drop(product_name, errors="ignore")
    return product_name, scores.head(top_n).index.tolist()


# --------------------------------------------------------------------------
# Data source — loaded directly from disk (no upload widget)
# --------------------------------------------------------------------------
# Change this path to point at your dataset.
DATA_PATH = "data/online_retail.csv"

st.sidebar.title("🛍️ Shopper Spectrum")
st.sidebar.caption("Customer Segmentation & Product Recommendation")

import os

if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
    st.sidebar.success(f"Loaded {len(df):,} rows from {DATA_PATH}")
else:
    st.sidebar.warning(f"'{DATA_PATH}' not found — using a generated demo dataset.")
    df = make_demo_data()

n_clusters = st.sidebar.slider("Number of customer segments (K)", 2, 6, 4)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview / EDA", "🧩 Customer Segmentation", "🎯 Product Recommendation"],
)

rfm, scaler, km, label_map = compute_rfm(df, n_clusters=n_clusters)
sim_df = build_similarity_matrix(df)

SEGMENT_COLORS = {
    "High-Value": "#2E7D32",
    "Regular": "#1565C0",
    "Occasional": "#F9A825",
    "At-Risk": "#C62828",
}

# --------------------------------------------------------------------------
# PAGE 1 — Overview / EDA
# --------------------------------------------------------------------------
if page == "📊 Overview / EDA":
    st.title("📊 Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    c2.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
    c3.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    c4.metric("Total Products", f"{df['Description'].nunique():,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        top_products = (
            df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        )
        fig = px.bar(
            top_products[::-1], orientation="h",
            title="Top 10 Best-Selling Products",
            labels={"value": "Units Sold", "Description": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_sales = (
            df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        )
        fig = px.bar(
            country_sales[::-1], orientation="h",
            title="Top 10 Countries by Revenue",
            labels={"value": "Revenue (£)", "Country": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
    fig = px.line(monthly, x="InvoiceDate", y="TotalPrice", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    most_active = df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending=False).head(10)
    fig = px.bar(
        most_active[::-1], orientation="h",
        title="Top 10 Most Active Customers (by # Orders)",
        labels={"value": "Orders", "CustomerID": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 2 — Customer Segmentation
# --------------------------------------------------------------------------
elif page == "🧩 Customer Segmentation":
    st.title("🧩 Customer Segmentation (RFM + K-Means)")

    tab1, tab2 = st.tabs(["🔍 Predict a Customer's Segment", "📈 Segment Explorer"])

    with tab1:
        st.subheader("Enter Customer RFM Values")
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Recency (days since last purchase)", min_value=0, value=30)
        frequency = c2.number_input("Frequency (number of orders)", min_value=1, value=5)
        monetary = c3.number_input("Monetary (total spend £)", min_value=0.0, value=500.0, step=10.0)

        if st.button("Predict Segment", type="primary"):
            segment = classify_new_customer(recency, frequency, monetary, rfm, scaler, km, label_map)
            color = SEGMENT_COLORS.get(segment, "#555555")
            st.markdown(
                f"""
                <div style="padding:1.2rem;border-radius:0.6rem;background-color:{color}22;
                            border:2px solid {color};text-align:center;">
                    <h3 style="color:{color};margin:0;">Predicted Segment: {segment}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Segment Distribution")
        col1, col2 = st.columns([1, 2])

        with col1:
            seg_counts = rfm["Segment"].value_counts()
            fig = px.pie(
                values=seg_counts.values, names=seg_counts.index,
                color=seg_counts.index, color_discrete_map=SEGMENT_COLORS,
                title="Customers per Segment",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                rfm, x="Recency", y="Monetary", size="Frequency", color="Segment",
                color_discrete_map=SEGMENT_COLORS, hover_data=["CustomerID"],
                title="Recency vs Monetary (bubble size = Frequency)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment Summary Statistics")
        summary = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
        st.dataframe(summary, use_container_width=True)

        st.subheader("Customer Table")
        st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 3 — Product Recommendation
# --------------------------------------------------------------------------
elif page == "🎯 Product Recommendation":
    st.title("🎯 Product Recommendation Engine")
    st.caption("Item-based collaborative filtering using cosine similarity.")

    product_list = sorted(df["Description"].unique().tolist())
    selected = st.selectbox("Choose or search a product", product_list)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get Recommendations", type="primary"):
        matched_name, recs = recommend_products(selected, sim_df, top_n=top_n)
        if not recs:
            st.warning("No similar products found.")
        else:
            st.success(f"Customers who bought **{matched_name}** also bought:")
            cols = st.columns(min(len(recs), 5))
            for i, prod in enumerate(recs):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""
                        <div style="padding:0.8rem;border-radius:0.5rem;border:1px solid #ddd;
                                    text-align:center;min-height:90px;">
                            <b>{prod}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

st.sidebar.divider()
st.sidebar.caption("Shopper Spectrum · Built with Streamlit")