import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.cluster import KMeans

# Title
st.title("Matiks User Behavior & Revenue Dashboard")

# Load data
uploaded_file = st.file_uploader("Upload user activity CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize and clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.title()

    # Rename known variants to match expected names
    column_renames = {
        "User_Id": "User_ID",
        "Signup": "Sign_Up",
        "Sign_Up_Date": "Sign_Up",
        "Signup_Date": "Sign_Up",
        "Total_Revenue_Usd": "Revenue"
    }
    df.rename(columns=column_renames, inplace=True)

    # Validate required columns
    required_cols = ["User_ID", "Last_Login", "Sign_Up", "Revenue"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}. Please check your CSV headers.")
        st.stop()

    # Parse dates
    df["Last_Login"] = pd.to_datetime(df["Last_Login"], errors="coerce")
    df["Sign_Up"] = pd.to_datetime(df["Sign_Up"], errors="coerce")
    df = df.dropna(subset=["Last_Login", "Sign_Up"])

    # Compute lifespan
    df["Lifespan"] = (df["Last_Login"] - df["Sign_Up"]).dt.days

    # DAU
    dau = df.groupby(df["Last_Login"].dt.date)["User_ID"].nunique().reset_index(name="DAU")
    dau["Login_Date"] = pd.to_datetime(dau["Last_Login"])

    # WAU
    df["Week"] = df["Last_Login"] - pd.to_timedelta(df["Last_Login"].dt.weekday, unit='d')
    wau = df.groupby("Week")["User_ID"].nunique().reset_index(name="WAU")

    # MAU
    df["Month"] = df["Last_Login"].dt.to_period("M").dt.to_timestamp()
    mau = df.groupby("Month")["User_ID"].nunique().reset_index(name="MAU")

    # Revenue trend
    revenue_trend = df.groupby("Month")["Revenue"].sum().reset_index()

    # Churn segmentation
    df = df[df["Lifespan"] >= 0]
    churn_labels = ['Same-Day', '≤ 7 Days', '≤ 30 Days']
    df['Churn_Group'] = pd.cut(df["Lifespan"], bins=[-1, 0, 7, 30], labels=churn_labels)

    total_users = df["User_ID"].nunique()
    total_revenue = df["Revenue"].sum()

    churn_data = df.groupby("Churn_Group").agg(
        User_Count=("User_ID", "count"),
        Avg_Revenue=("Revenue", "mean"),
        Total_Revenue=("Revenue", "sum")
    ).reset_index()

    churn_data["User_%"] = (churn_data["User_Count"] / total_users * 100).round(2)
    churn_data["Revenue_%"] = (churn_data["Total_Revenue"] / total_revenue * 100).round(2)

    # Loyalty bands
    bins = [0, 100, 300, 500, df["Lifespan"].max() + 1]
    labels = ['<100 days', '100–300 days', '300–500 days', '500+ days']
    df['Loyalty_Band'] = pd.cut(df['Lifespan'], bins=bins, labels=labels, right=False)

    avg_rev_band = df.groupby("Loyalty_Band")["Revenue"].mean().reset_index()
    total_rev_band = df.groupby("Loyalty_Band")["Revenue"].sum().reset_index()

    # Clustering
    clustering_data = df[["Lifespan", "Revenue"]].dropna()
    kmeans = KMeans(n_clusters=4, random_state=42).fit(clustering_data)
    clustering_data["Cluster"] = kmeans.labels_

    # Active Users Plots
    st.subheader("Active Users Overview")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(dau["Login_Date"], dau["DAU"], marker='o', color="steelblue")
    axs[0].set_title("Daily Active Users (DAU)")
    axs[0].set_ylabel("Users")

    axs[1].plot(wau["Week"], wau["WAU"], marker='o', color="darkorange")
    axs[1].set_title("Weekly Active Users (WAU)")
    axs[1].set_ylabel("Users")

    axs[2].plot(mau["Month"], mau["MAU"], marker='o', color="seagreen")
    axs[2].set_title("Monthly Active Users (MAU)")
    axs[2].set_ylabel("Users")
    axs[2].set_xlabel("Date")

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for ax in axs:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)

    # Revenue Trend Plot
    st.subheader("Revenue Trends Over Time")
    fig_rev, ax_rev = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=revenue_trend, x="Month", y="Revenue", marker='o', ax=ax_rev, color="green")
    ax_rev.set_title("Monthly Revenue")
    ax_rev.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rev.tick_params(axis='x', rotation=45)
    st.pyplot(fig_rev)

    # Churn Segments Plot
    st.subheader("Early Churn Segments")
    fig_churn, ax1 = plt.subplots(figsize=(10, 6))
    bar = ax1.bar(churn_data["Churn_Group"], churn_data["User_%"], color='skyblue', label='User %')
    ax1.set_ylabel("User %", color='blue')
    ax1.set_xlabel("Churn Risk Group")
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(churn_data["Churn_Group"], churn_data["Avg_Revenue"], color='red', marker='o', linewidth=2, label='Avg Revenue')
    ax2.set_ylabel("Avg Revenue (USD)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    for i, rect in enumerate(bar):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, height, f"{churn_data['Revenue_%'][i]}%", ha='center', va='bottom', fontsize=10)

    ax1.set_title("Churn Segments: User %, Revenue Share, Avg Revenue")
    st.pyplot(fig_churn)

    # Loyalty Band Plot
    st.subheader("Loyalty Band Revenue")
    fig_loyalty, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=avg_rev_band, x="Loyalty_Band", y="Revenue", ax=ax)
    ax.set_title("Average Revenue by Loyalty Band")
    st.pyplot(fig_loyalty)

    # Clustering Plot
    st.subheader("User Clustering: Lifespan vs Revenue")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=clustering_data, x="Lifespan", y="Revenue", hue="Cluster", palette="deep", ax=ax_cluster)
    ax_cluster.set_title("KMeans Clustering")
    st.pyplot(fig_cluster)

    st.success("File uploaded and analyzed successfully!")
