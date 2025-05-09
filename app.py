import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("delay_model.pkl")
le_airline = joblib.load("le_airline.pkl")
le_origin = joblib.load("le_origin.pkl")
le_dest = joblib.load("le_dest.pkl")

# UI
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

# For dropdown options (you can also hardcode these)
sample_df = pd.read_csv("Flight_data_part_1.csv")  # Any part is fine
month_options = sorted(sample_df['Month'].dropna().unique())
airline_options = sorted(sample_df['AIRLINE'].dropna().unique())

selected_month = st.selectbox("Select Month", month_options)
selected_airline = st.selectbox("Select Airline", airline_options)
origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL, ORD)").upper()
destination_input = st.text_input("Enter Destination Airport Code (e.g., LAX, JFK)").upper()

# Validate input
if origin_input and destination_input:
    try:
        input_data = pd.DataFrame([{
            'Month': selected_month,
            'AIRLINE_ENC': le_airline.transform([selected_airline])[0],
            'ORIGIN_ENC': le_origin.transform([origin_input])[0],
            'DEST_ENC': le_dest.transform([destination_input])[0]
        }])
        delay_proba = model.predict_proba(input_data)[0][1]
        st.success(f"📊 Estimated Delay Probability: **{delay_proba * 100:.2f}%**")
    except ValueError as e:
        st.error("❌ One or more inputs not recognized by the model.")
