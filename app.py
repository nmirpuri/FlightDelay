import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import gdown
import requests
from io import StringIO


def load_and_preprocess_data():
    try:
        file_id = "1LpVqLHQVmIlAnSqeEcFSK6R1v5P5_WEW"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        output = "Flight_data.csv"

        # Download from Google Drive using gdown
        gdown.download(url, output, quiet=False)

        # Load the actual CSV
        df = pd.read_csv(output)

        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return None

df = load_and_preprocess_data()
if df is None:
    st.error("Failed to load dataset.")
    st.stop()

# === Train Model ===
@st.cache_resource
def train_model(df):
    le_airline = LabelEncoder()
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()

    df['AIRLINE'] = le_airline.fit_transform(df['AIRLINE'])
    df['ORIGIN'] = le_origin.fit_transform(df['ORIGIN'])
    df['DEST'] = le_dest.fit_transform(df['DEST'])
    X = df[['AIRLINE', 'Month', 'ORIGIN', 'DEST']]
    y = df['Delayed']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    encoders = {
        'AIRLINE': le_airline,
        'ORIGIN': le_origin,
        'DEST': le_dest
    }

    return model, encoders


# === App Interface ===
st.title("Flight Delay Predictor ✈️")
st.write("Enter flight details to estimate the probability of a delay:")

df, encoders = load_and_preprocess_data()
model = train_model(df)

# Collect user input
airline = st.selectbox("AIRLINE", encoders['AIRLINE'].classes_)
month = st.selectbox("Month", encoders['Month'].classes_)
origin = st.selectbox("ORIGIN Airport", encoders['ORIGIN'].classes_)
destination = st.selectbox("DEST Airport", encoders['DEST'].classes_)

# Encode inputs
input_data = {
    'AIRLINE': encoders['AIRLINE'].transform([airline])[0],
    'Month': encoders['Month'].transform([month])[0],
    'ORIGIN': encoders['ORIGIN'].transform([origin])[0],
    'DEST': encoders['DEST'].transform([destination])[0]
}

# Predict
if st.button("Predict Delay"):
    prediction = model.predict_proba([[input_data['AIRLINE'], input_data['Month'],
                                        input_data['ORIGIN'], input_data['DEST']]])[0][1]
    st.success(f"Estimated Probability of Delay: **{prediction * 100:.2f}%**")
