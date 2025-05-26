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
    df.rename(columns={"User ID": "User_ID"}, inplace=True)

    # Ensure datetime columns are parsed correctly
    df["Last_Login"] = pd.to_datetime(df["Last_Login"], errors="coerce")
    df["Sign_Up"] = pd.to_datetime(df["Sign_Up"], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["Last_Login", "Sign_Up"])

    # Calculate DAU
    dau = df.groupby(df["Last_Login"].dt.date)["User_ID"].nunique().reset_index(name="DAU")
    dau["Login_Date"] = pd.to_datetime(dau["Last_Login"])

    # Calculate WAU
    df["Week"] = df["Last_Login"] - pd.to_timedelta(df["Last_Login"].dt.weekday, unit='d')
    wau = df.groupby("Week")["User_ID"].nunique().reset_index(name="WAU")

    # Calculate MAU
    df["Month"] = df["Last_Login"].dt.to_period("M").dt.to_timestamp()
    mau = df.groupby("Month")["User_ID"].nunique().reset_index(name="MAU")

    # Revenue trend
    revenue_trend = df.groupby("Month")["Revenue"].sum().reset_index()

    # Revenue by segment breakdown
    rev_segment = df.groupby("Tier")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
    rev_device = df.groupby("Device_Type")["Revenue"].sum().reset_index()
    rev_game = df.groupby("Game_Mode")["Revenue"].sum().reset_index()
    rev_user_segment = df.groupby("User_Segment")["Revenue"].sum().reset_index()

    # User lifespan in days
    df["Lifespan"] = (df["Last_Login"] - df["Sign_Up"]).dt.days

    # Clustering: Frequency vs Revenue
    clustering_data = df[["Lifespan", "Revenue"]].dropna()
    kmeans = KMeans(n_clusters=4, random_state=42).fit(clustering_data)
    clustering_data["Cluster"] = kmeans.labels_

    # Plot charts
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

    st.subheader("Revenue Trends Over Time")
    fig_rev, ax_rev = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=revenue_trend, x="Month", y="Revenue", marker='o', ax=ax_rev, color="green")
    ax_rev.set_title("Monthly Revenue")
    ax_rev.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rev.tick_params(axis='x', rotation=45)
    st.pyplot(fig_rev)

    st.subheader("Revenue by Segment")
    col1, col2 = st.columns(2)
    with col1:
        st.write("By Tier")
        st.dataframe(rev_segment)
    with col2:
        st.write("By Device Type")
        st.dataframe(rev_device)

    col3, col4 = st.columns(2)
    with col3:
        st.write("By Game Mode")
        st.dataframe(rev_game)
    with col4:
        st.write("By User Segment")
        st.dataframe(rev_user_segment)

    st.subheader("User Clusters: Frequency vs Revenue")
    fig_clust, ax_clust = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=clustering_data, x="Lifespan", y="Revenue", hue="Cluster", palette="deep", ax=ax_clust)
    ax_clust.set_title("User Clustering")
    st.pyplot(fig_clust)

    st.subheader("Cohort Analysis: User Growth and Revenue Trends")
    df["Signup_Month"] = df["Sign_Up"].dt.to_period("M").dt.to_timestamp()
    cohort_data = df.groupby("Signup_Month").agg(
        User_Count=("User_ID", "count"),
        Avg_Revenue=("Revenue", "mean")
    ).reset_index()

    fig_cohort_mix, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(cohort_data["Signup_Month"], cohort_data["User_Count"], color='skyblue', label='User Count')
    ax1.set_ylabel("User Count", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.plot(cohort_data["Signup_Month"], cohort_data["Avg_Revenue"], color='orange', marker='o', label='Avg Revenue/User')
    ax2.set_ylabel("Avg Revenue/User (USD)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax1.set_title("Cohort Analysis: User Growth and Revenue Trends")
    ax1.set_xlabel("Cohort Month")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig_cohort_mix)

    st.subheader("Revenue Breakdown Visualizations")
    fig_all, axs_all = plt.subplots(3, 1, figsize=(10, 15))
    sns.barplot(data=rev_device, x="Device_Type", y="Revenue", ax=axs_all[0])
    axs_all[0].set_title("Revenue by Device Type")

    sns.barplot(data=rev_segment, x="Tier", y="Revenue", ax=axs_all[1], palette="pastel")
    axs_all[1].set_title("Revenue by Subscription Tier")

    sns.barplot(data=rev_game, x="Game_Mode", y="Revenue", ax=axs_all[2])
    axs_all[2].set_title("Revenue by Preferred Game Mode")
    axs_all[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig_all)

    st.subheader("Revenue vs User Loyalty")
    bins = [0, 100, 300, 500, df["Lifespan"].max() + 1]
    labels = ['<100 days', '100–300 days', '300–500 days', '500+ days']
    df['Loyalty_Band'] = pd.cut(df['Lifespan'], bins=bins, labels=labels, right=False)

    avg_rev_band = df.groupby("Loyalty_Band")["Revenue"].mean().reset_index()
    fig_avg_band, ax_avg_band = plt.subplots(figsize=(10, 5))
    sns.barplot(data=avg_rev_band, x="Loyalty_Band", y="Revenue", ax=ax_avg_band)
    ax_avg_band.set_title("Average Revenue by User Loyalty Band")
    ax_avg_band.set_ylabel("Average Revenue (USD)")
    ax_avg_band.set_xlabel("Loyalty Band (Lifespan)")
    st.pyplot(fig_avg_band)

    total_rev_band = df.groupby("Loyalty_Band")["Revenue"].sum().reset_index()
    fig_total_band, ax_total_band = plt.subplots(figsize=(10, 5))
    sns.barplot(data=total_rev_band, x="Loyalty_Band", y="Revenue", ax=ax_total_band)
    ax_total_band.set_title("Total Revenue by User Loyalty Band")
    ax_total_band.set_ylabel("Total Revenue (USD)")
    ax_total_band.set_xlabel("Loyalty Band (Lifespan)")
    st.pyplot(fig_total_band)

    st.subheader("Early Churn Segments: User Count, Revenue Share, and Avg Revenue")
    df = df[df["Lifespan"] >= 0]  # Ensure lifespan is non-negative
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

    fig_churn, ax1 = plt.subplots(figsize=(10, 6))
    bar = ax1.bar(churn_data["Churn_Group"], churn_data["User_%"], color='skyblue', label='User %')
    ax1.set_ylabel("User %", color='blue')
    ax1.set_xlabel("Churn Risk Group")
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    line = ax2.plot(churn_data["Churn_Group"], churn_data["Avg_Revenue"], color='red', marker='o', linewidth=2, label='Avg Revenue')
    ax2.set_ylabel("Average Revenue (USD)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    for i, rect in enumerate(bar):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, height, f"{churn_data['Revenue_%'][i]}%", ha='center', va='bottom', fontsize=10, color='black')

    ax1.set_title("Early Churn Segments: User %, Revenue Share, and Avg Revenue")
    st.pyplot(fig_churn)

    st.subheader("High-Value vs High-Retention Users: Key Characteristics")
    top_value = df.sort_values(by="Revenue", ascending=False).head(1000)
    top_retention = df.sort_values(by="Lifespan", ascending=False).head(1000)

    fig_hr, axs_hr = plt.subplots(2, 3, figsize=(18, 10))
    bar_colors = {
        "High-Value": "skyblue",
        "High-Retention": "lightgreen"
    }

    sns.countplot(data=top_value, x="Tier", ax=axs_hr[0, 0], color=bar_colors["High-Value"])
    axs_hr[0, 0].set_title("High-Value: Subscription Tiers")

    sns.countplot(data=top_value, x="Device_Type", ax=axs_hr[0, 1], color=bar_colors["High-Value"])
    axs_hr[0, 1].set_title("High-Value: Devices")

    sns.countplot(data=top_value, x="Game_Mode", ax=axs_hr[0, 2], color=bar_colors["High-Value"])
    axs_hr[0, 2].set_title("High-Value: Game Modes")

    sns.countplot(data=top_retention, x="Tier", ax=axs_hr[1, 0], color=bar_colors["High-Retention"])
    axs_hr[1, 0].set_title("High-Retention: Subscription Tiers")

    sns.countplot(data=top_retention, x="Device_Type", ax=axs_hr[1, 1], color=bar_colors["High-Retention"])
    axs_hr[1, 1].set_title("High-Retention: Devices")

    sns.countplot(data=top_retention, x="Game_Mode", ax=axs_hr[1, 2], color=bar_colors["High-Retention"])
    axs_hr[1, 2].set_title("High-Retention: Game Modes")

    for ax in axs_hr.flat:
        ax.set_ylabel("User Count")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("High-Value vs High-Retention Users: Key Characteristics", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig_hr)
else:
    st.info("Please upload a CSV file with columns including 'User ID', 'Last_Login', 'Sign_Up', 'Revenue', 'Tier', 'Device_Type', 'Game_Mode', 'User_Segment'")
