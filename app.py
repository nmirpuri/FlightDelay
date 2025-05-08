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
        url = f"https://drive.google.com/uc?id={file_id}"

        # Use requests to download the file
        response = requests.get(url)
        
        # Check for successful download
        if response.status_code == 200:
            # Convert content to StringIO, which is like a file object
            data = StringIO(response.text)

            # Read into pandas dataframe
            df = pd.read_csv(data)

            # DEBUG: Show shape and columns
            print("CSV Loaded: ", df.shape)
            print("Columns: ", df.columns)
            
            # Your preprocessing steps here...
            
            return df, {}
        else:
            print(f"Failed to download file, status code: {response.status_code}")
            return None, None
        
    except Exception as e:
        print("❌ ERROR in load_and_preprocess_data():", e)
        return None, None

    # Create binary target column
    df['Delayed'] = df['Delay'].apply(lambda x: 1 if x > 0 else 0)

    # Encode categorical columns
    encoders = {}
    for col in ['Airline', 'Month', 'Origin', 'Destination']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

# === Train Model ===
@st.cache_resource
def train_model(df):
    X = df[['AIRLINE', 'Month', 'ORIGIN', 'DEST']]
    y = df['Delayed']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model

# === App Interface ===
st.title("Flight Delay Predictor ✈️")
st.write("Enter flight details to estimate the probability of a delay:")

df, encoders = load_and_preprocess_data()
model = train_model(df)

# Collect user input
airline = st.selectbox("Airline", encoders['Airline'].classes_)
month = st.selectbox("Month", encoders['Month'].classes_)
origin = st.selectbox("Origin Airport", encoders['Origin'].classes_)
destination = st.selectbox("Destination Airport", encoders['Destination'].classes_)

# Encode inputs
input_data = {
    'Airline': encoders['Airline'].transform([airline])[0],
    'Month': encoders['Month'].transform([month])[0],
    'Origin': encoders['Origin'].transform([origin])[0],
    'Destination': encoders['Destination'].transform([destination])[0]
}

# Predict
if st.button("Predict Delay"):
    prediction = model.predict_proba([[input_data['Airline'], input_data['Month'],
                                        input_data['Origin'], input_data['Destination']]])[0][1]
    st.success(f"Estimated Probability of Delay: **{prediction * 100:.2f}%**")
