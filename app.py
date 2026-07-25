import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
st.set_page_config(
    page_title="Traffic Pattern Analysis",
    page_icon="🚦",
    layout="wide"
)
st.sidebar.title("🚦 Traffic Pattern Analysis")

st.sidebar.write("""
### Project Details

- Algorithm: K-Means Clustering
- Machine Learning Type: Unsupervised Learning
- Features Used:
  - Traffic Volume
  - Temperature
  - Rain
  - Snow
  - Clouds
  - Hour
  - Day
  - Month
""")

st.title("🚦 Traffic Pattern Analysis using K-Means")

uploaded_file = st.file_uploader(
    "traffic_pattern(sample).csv",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully!")

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        format="%d-%m-%Y %H:%M"
    )

    data["hour"] = data["date_time"].dt.hour
    data["day"] = data["date_time"].dt.dayofweek
    data["month"] = data["date_time"].dt.month

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Clusters", 3)

    st.subheader("Elbow Method")

    X = data[[
        "traffic_volume",
        "temperature",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "hour",
        "day",
        "month"
    ]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []

    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    st.subheader("Choose Number of Clusters")

    clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    model = KMeans(
        n_clusters=clusters,
        random_state=42
    )

    data["Cluster"] = model.fit_predict(X_scaled)
    st.subheader("Traffic Cluster Visualization")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    scatter = ax2.scatter(
        data["traffic_volume"],
        data["temperature"],
        c=data["Cluster"],
        cmap="viridis"
    )

    plt.colorbar(scatter)

    ax2.set_xlabel("Traffic Volume")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Traffic Clusters")

    st.pyplot(fig2)
    st.subheader("Cluster Statistics")

    cluster_summary = data.groupby("Cluster").mean(numeric_only=True)

    st.dataframe(cluster_summary)
    st.subheader("Traffic Volume Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.hist(data["traffic_volume"], bins=30)

    ax3.set_xlabel("Traffic Volume")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Traffic Volume Distribution")

    st.pyplot(fig3)
    st.subheader("Average Traffic by Hour")

    hourly = data.groupby("hour")["traffic_volume"].mean()

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    ax4.plot(hourly, marker="o")

    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Average Traffic Volume")
    ax4.set_title("Traffic by Hour")

    st.pyplot(fig4)
    st.subheader("Download Results")

    csv = data.to_csv(index=False)

    st.download_button(
        label="Download Clustered CSV",
        data=csv,
        file_name="traffic_cluster_output.csv",
        mime="text/csv"
    )
    st.markdown("---")

    st.header("About This Project")

    st.write("""
    This application analyzes traffic patterns using the K-Means clustering algorithm.

    Workflow:
    1. Upload traffic dataset.
    2. Perform feature engineering.
    3. Standardize features.
    4. Determine optimal clusters using the Elbow Method.
    5. Cluster the traffic data.
    6. Visualize and analyze the results.
    """)