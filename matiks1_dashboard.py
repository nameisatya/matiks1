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

    # Proceed with rest of the app logic...
    # (This section would continue with calculations and visualizations)
    st.success("File uploaded and processed successfully!")
