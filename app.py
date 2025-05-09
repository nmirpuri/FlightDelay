import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")

st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

# --- Load Data ---
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
st.write(f"✅ Loaded {len(df)} rows.")

# --- Preprocessing ---
# Keep only relevant columns and drop rows with any missing values
required_columns = ['Month', 'AIRLINE', 'ORIGIN', 'DEST', 'Delayed']
df = df[required_columns].dropna()

# Ensure correct types
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df['Delayed'] = pd.to_numeric(df['Delayed'], errors='coerce')
df.dropna(inplace=True)

# Label encode
le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['AIRLINE_ENC'] = le_airline.fit_transform(df['AIRLINE'])
df['ORIGIN_ENC'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_ENC'] = le_dest.fit_transform(df['DEST'])

# Final feature matrix
X = df[['Month', 'AIRLINE_ENC', 'ORIGIN_ENC', 'DEST_ENC']]
y = df['Delayed']

# Final check for numeric dtype and alignment
X = X.apply(pd.to_numeric)
y = y.astype(int)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- UI Inputs ---
month_options = sorted(df['Month'].unique())
airline_options = sorted(df['AIRLINE'].unique())

selected_month = st.selectbox("Select Month", month_options)
selected_airline = st.selectbox("Select Airline", airline_options)

origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL, ORD)").upper()
destination_input = st.text_input("Enter Destination Airport Code (e.g., LAX, JFK)").upper()

valid_origin = origin_input in df['ORIGIN'].unique()
valid_destination = destination_input in df['DEST'].unique()

if origin_input and not valid_origin:
    st.error("❌ Origin not found in dataset.")
if destination_input and not valid_destination:
    st.error("❌ Destination not found in dataset.")

# --- Prediction ---
if st.button("Predict Delay Probability") and valid_origin and valid_destination:
    try:
        input_data = pd.DataFrame([{
            'Month': selected_month,
            'AIRLINE_ENC': le_airline.transform([selected_airline])[0],
            'ORIGIN_ENC': le_origin.transform([origin_input])[0],
            'DEST_ENC': le_dest.transform([destination_input])[0]
        }])

        probability = model.predict_proba(input_data)[0][1] * 100
        st.success(f"✈️ Estimated Probability of Delay: **{probability:.2f}%**")
    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
