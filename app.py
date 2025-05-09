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
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
st.write(f"✅ Loaded {len(df)} rows.")

# --- Basic Preprocessing ---
df = df[['MONTH', 'AIRLINE', 'ORIGIN', 'DEST', 'DEP_DELAY', 'ARR_DELAY']]
df.dropna(inplace=True)

# Define delay as arrival delay > 15 mins
df['DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)

# Label encode categorical variables
le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['AIRLINE_ENC'] = le_airline.fit_transform(df['AIRLINE'])
df['ORIGIN_ENC'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_ENC'] = le_dest.fit_transform(df['DEST'])

X = df[['MONTH', 'AIRLINE_ENC', 'ORIGIN_ENC', 'DEST_ENC']]
y = df['DELAYED']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- UI Components ---
month_options = sorted(df['MONTH'].dropna().unique())
selected_month = st.selectbox("Select Month", month_options)

airline_options = sorted(df['AIRLINE'].dropna().unique())
selected_airline = st.selectbox("Select Airline", airline_options)

origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL, ORD)").upper()
destination_input = st.text_input("Enter Destination Airport Code (e.g., LAX, JFK)").upper()

valid_origin = origin_input in df['ORIGIN'].unique()
valid_destination = destination_input in df['DEST'].unique()

if origin_input and not valid_origin:
    st.error("❌ Origin not found in dataset.")
if destination_input and not valid_destination:
    st.error("❌ Destination not found in dataset.")

# --- Predict on Input ---
if st.button("Predict Delay Probability") and valid_origin and valid_destination:
    try:
        input_data = pd.DataFrame([{
            'MONTH': selected_month,
            'AIRLINE_ENC': le_airline.transform([selected_airline])[0],
            'ORIGIN_ENC': le_origin.transform([origin_input])[0],
            'DEST_ENC': le_dest.transform([destination_input])[0]
        }])
        
        prob = model.predict_proba(input_data)[0][1] * 100
        st.success(f"✈️ Estimated Probability of Delay: **{prob:.2f}%**")
    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
