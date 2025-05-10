import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit UI setup ---
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

# --- Load data ---
files = [f"Flight_data_part_{i}.csv" for i in range(1, 7)]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
st.write(f"✅ Loaded {len(df)} rows.")

# --- Encode categorical columns ---
encoders = {}
for col in ['Month', 'AIRLINE', 'ORIGIN', 'DEST']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- Train model ---
X = df[['Month', 'AIRLINE', 'ORIGIN', 'DEST']]
y = df['Delayed']
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# --- UI Inputs ---
# Dropdowns
month_label = st.selectbox("Select Month", sorted(encoders['Month'].classes_))
airline_label = st.selectbox("Select Airline", sorted(encoders['AIRLINE'].classes_))

# Text inputs
origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL, ORD)").upper()
dest_input = st.text_input("Enter Destination Airport Code (e.g., LAX, JFK)").upper()

# Validate origin/destination
origin_valid = origin_input in encoders['ORIGIN'].classes_
dest_valid = dest_input in encoders['DEST'].classes_

if origin_input and not origin_valid:
    st.error("❌ Origin not found in dataset.")
if dest_input and not dest_valid:
    st.error("❌ Destination not found in dataset.")

# --- Prediction ---
if st.button("Predict Delay Probability"):
    if origin_valid and dest_valid:
        input_data = pd.DataFrame([{
            'Month': encoders['Month'].transform([month_label])[0],
            'AIRLINE': encoders['AIRLINE'].transform([airline_label])[0],
            'ORIGIN': encoders['ORIGIN'].transform([origin_input])[0],
            'DEST': encoders['DEST'].transform([dest_input])[0],
        }])
        prob = model.predict_proba(input_data)[0][1]
        st.success(f"🚨 Estimated Delay Probability: **{prob * 100:.2f}%**")
    else:
        st.warning("⚠️ Please enter valid airport codes to predict.")
