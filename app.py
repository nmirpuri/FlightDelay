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
        # Your public Google Drive file ID
        file_id = "1LpVqLHQVmIlAnSqeEcFSK6R1v5P5_WEW"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Fetch the file content
        response = requests.get(url)
        
        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data)

            print("✅ CSV Loaded:", df.shape)
            print("✅ Columns:", df.columns.tolist())

            # Basic check for required columns
            required_cols = {'Month', 'ORIGIN', 'DEST', 'AIRLINE', 'Delayed'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

            # Encode categorical columns
            encoders = {}
            for col in ['AIRLINE', 'Month', 'ORIGIN', 'DEST']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le

            return df, encoders
        else:
            print("❌ Failed to fetch file. Status code:", response.status_code)
            return None, None

    except Exception as e:
        print("❌ ERROR in load_and_preprocess_data():", e)
        return None, None

df, encoders = load_and_preprocess_data()

if df is not None:
    st.success("✅ Data loaded successfully!")
    st.write(df.head())
else:
    st.error("❌ Failed to load data.")

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
