import streamlit as st
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")

st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

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

# Check result
st.write(f"✅ Loaded {len(df)} rows.")
st.write(df.head())





   
