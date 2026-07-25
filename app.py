import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Traffic Pattern Analysis",
    page_icon="🚦",
    layout="wide"
)


st.sidebar.title("🚦 Traffic Pattern Analysis")

st.sidebar.markdown("""
### Project Details

**Algorithm:** K-Means Clustering

**Machine Learning Type:** Unsupervised Learning

### Features Used
- Traffic Volume
- Temperature
- Rain
- Snow
- Clouds
- Hour
- Day
- Month
""")


st.title("🚦 Traffic Pattern Analysis using K-Means Clustering")

uploaded_file = st.file_uploader(
    "📂 Upload Traffic Dataset (.csv)",
    type=["csv"],
    help="Upload a traffic dataset in CSV format."
)


if uploaded_file is not None:

    # Read Dataset
    data = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully!")

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        format="%d-%m-%Y %H:%M"
    )

    data["hour"] = data["date_time"].dt.hour
    data["day"] = data["date_time"].dt.dayofweek
    data["month"] = data["date_time"].dt.month

 
    st.subheader("📋 Dataset Preview")

    with st.expander("View Dataset"):
        st.dataframe(data.head(10), use_container_width=True)

   
    X = data[
        [
            "traffic_volume",
            "temperature",
            "rain_1h",
            "snow_1h",
            "clouds_all",
            "hour",
            "day",
            "month",
        ]
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    st.subheader("📈 Elbow Method")

    wcss = []

    for i in range(1, 11):
        model = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

    st.subheader("🎯 Choose Number of Clusters")

    clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Rows", data.shape[0])
    col2.metric("📋 Columns", data.shape[1])
    col3.metric("🎯 Clusters", clusters)

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = model.fit_predict(X_scaled)

    st.subheader("📌 Cluster Insights")

    for cluster in sorted(data["Cluster"].unique()):

        cluster_data = data[data["Cluster"] == cluster]

        st.markdown(f"### Cluster {cluster}")
        st.write(f"**Records:** {len(cluster_data)}")
        st.write(
            f"**Average Traffic Volume:** {cluster_data['traffic_volume'].mean():.2f}"
        )
        st.write(
            f"**Average Temperature:** {cluster_data['temperature'].mean():.2f} °C"
        )
        st.write(
            f"**Average Clouds:** {cluster_data['clouds_all'].mean():.2f}%"
        )

    st.subheader("📊 Traffic Cluster Visualization")

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

    st.subheader("📑 Cluster Statistics")

    cluster_summary = data.groupby("Cluster").mean(numeric_only=True)

    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("📊 Traffic Volume Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.hist(
        data["traffic_volume"],
        bins=30
    )

    ax3.set_xlabel("Traffic Volume")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Traffic Volume Distribution")

    st.pyplot(fig3)

    st.subheader("🕒 Average Traffic by Hour")

    hourly = data.groupby("hour")["traffic_volume"].mean()

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    ax4.plot(
        hourly.index,
        hourly.values,
        marker="o"
    )

    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Average Traffic Volume")
    ax4.set_title("Average Traffic by Hour")

    st.pyplot(fig4)

    st.subheader("⬇ Download Results")

    csv = data.to_csv(index=False)

    st.download_button(
        label="📥 Download Clustered CSV",
        data=csv,
        file_name="traffic_cluster_output.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.header("📖 About This Project")

    st.markdown("""
This dashboard analyzes traffic patterns using the **K-Means Clustering** algorithm.

### Features
- 📂 Upload your own CSV dataset
- 📈 Visualize the Elbow Method
- 🎯 Select the number of clusters
- 📊 Explore cluster insights
- 📉 Analyze traffic distribution
- 🕒 Visualize hourly traffic patterns
- 📥 Download clustered dataset
""")

else:
    st.info("👆 Please upload a CSV dataset to begin the analysis.")
