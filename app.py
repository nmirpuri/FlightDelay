import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/nmirpuri/FlightDelay/refs/heads/main/Backy.avif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .main > div {
        background-color: rgba(255, 255, 255, 0.8);  /* Light overlay behind content */
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True



)




corner_image_css = """
<style>
.corner-image {
    position: fixed;
    bottom: 0px;
    left: 0px;
    width: 250px;
    z-index: 100;
}
</style>

<img src="https://raw.githubusercontent.com/nmirpuri/FlightDelay/refs/heads/main/6485090.png" class="corner-image">
"""

st.markdown(corner_image_css, unsafe_allow_html=True)



st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")


# Load and sample data
files = [
    "Flight_data_part_1.csv",
    "Flight_data_part_2.csv",
    "Flight_data_part_3.csv",
    "Flight_data_part_4.csv",
    "Flight_data_part_5.csv",
    "Flight_data_part_6.csv"
]

df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
df = df.sample(n=200_000, random_state=42).reset_index(drop=True)



# Cache model training
@st.cache_resource
def train_model(data):
    data = data.dropna(subset=['Delayed'])
    data = data[data['Delayed'].isin([0, 1])]

    for col in ['Month', 'AIRLINE', 'ORIGIN', 'DEST']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data[['Month', 'AIRLINE', 'ORIGIN', 'DEST']]
    y = data['Delayed']

    model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, le

model, le = train_model(df)

# --- UI ---

month_options = sorted(df['Month'].dropna().unique())
selected_month = st.selectbox("Select Month", month_options)

airline_options = sorted(df['AIRLINE'].dropna().unique())
selected_airline = st.selectbox("Select Airline", airline_options)

origin_input = st.text_input("Enter Origin Airport Code (e.g., ATL)").upper()
destination_input = st.text_input("Enter Destination Airport Code (e.g., LAX)").upper()

if origin_input and destination_input:
    if origin_input not in df['ORIGIN'].unique():
        st.error("❌ Origin not found.")
    elif destination_input not in df['DEST'].unique():
        st.error("❌ Destination not found.")
    else:
        # Encode inputs
        input_df = pd.DataFrame({
            'Month': [selected_month],
            'AIRLINE': [selected_airline],
            'ORIGIN': [origin_input],
            'DEST': [destination_input]
        })

        for col in input_df.columns:
            input_df[col] = le.fit(df[col]).transform(input_df[col])

        prob = model.predict_proba(input_df)[0][1]
        st.success(f"✈️ Probability of delay: **{prob * 100:.2f}%**")
