import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")

st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

# Load and merge all 6 parts
files = [
    "Flight_data_part_1.csv",
    "Flight_data_part_2.csv",
    "Flight_data_part_3.csv",
    "Flight_data_part_4.csv",
    "Flight_data_part_5.csv",
    "Flight_data_part_6.csv"
]

# Read and concatenate
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# Show data loaded
st.write(f"✅ Loaded {len(df)} rows.")
st.write(df.head())

# --- UI COMPONENTS ---

# Dropdown for Month (sorted)
month_options = sorted(df['Month'].dropna().unique())
selected_month = st.selectbox("Select Month", month_options)

# Dropdown for Airline (sorted)
airline_options = sorted(df['AIRLINE'].dropna().unique())
selected_airline = st.selectbox("Select Airline", airline_options)

# Text input for Origin
origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL, ORD)").upper()
if origin_input and origin_input not in df['ORIGIN'].unique():
    st.error("❌ Origin not found in dataset.")

# Text input for Destination
destination_input = st.text_input("Enter Destination Airport Code (e.g., LAX, JFK)").upper()
if destination_input and destination_input not in df['DEST'].unique():
    st.error("❌ Destination not found in dataset.")
