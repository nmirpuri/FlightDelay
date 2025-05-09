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
print(f"✅ Loaded {len(df)} rows.")
print(df.head())



st.subheader("🔍 Predict Delay")
with st.form("flight_form"):
    airline_label = st.selectbox("Airline", encoders['AIRLINE'].classes_)
    month_label = st.selectbox("Month", encoders['Month'].classes_)
    origin_label = st.selectbox("Origin Airport", encoders['ORIGIN'].classes_)
    dest_label = st.selectbox("Destination Airport", encoders['DEST'].classes_)
    submit = st.form_submit_button("Predict Delay")

if submit:
    try:
        input_df = pd.DataFrame([{
            'Month': encoders['Month'].transform([month_label])[0],
            'ORIGIN': encoders['ORIGIN'].transform([origin_label])[0],
            'DEST': encoders['DEST'].transform([dest_label])[0],
            'AIRLINE': encoders['AIRLINE'].transform([airline_label])[0]
        }])

     
