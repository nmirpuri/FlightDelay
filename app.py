import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")

st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

@st.cache_data
def load_data():
    parts = []
    for i in range(1, 7):
        try:
            part = pd.read_csv(f"Flight_data_part_{i}.csv")
            parts.append(part)
        except Exception as e:
            st.error(f"Failed to load Flight_data_part_{i}.csv: {e}")
            return None
    df = pd.concat(parts, ignore_index=True)
    return df

@st.cache_data
def preprocess_data(df):
    encoders = {}
    for col in ['Month', 'ORIGIN', 'DEST', 'AIRLINE']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

@st.cache_resource
def train_model(df):
    X = df[['Month', 'ORIGIN', 'DEST', 'AIRLINE']]
    y = df['Delayed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# Load and train
df = load_data()
if df is not None:
    st.success("✅ Data loaded and combined successfully.")
    df, encoders = preprocess_data(df)
    model = train_model(df)

    # User input form
    st.subheader("🔍 Predict Delay")
    with st.form("flight_form"):
        airline = st.selectbox("Airline", df['AIRLINE'].unique())
        month = st.selectbox("Month", df['Month'].unique())
        origin = st.selectbox("Origin Airport", df['ORIGIN'].unique())
        dest = st.selectbox("Destination Airport", df['DEST'].unique())
        submit = st.form_submit_button("Predict Delay")

    if submit:
        # Encode user input
        try:
            input_df = pd.DataFrame([{
                'Month': month,
                'ORIGIN': origin,
                'DEST': dest,
                'AIRLINE': airline
            }])
            for col in ['Month', 'ORIGIN', 'DEST', 'AIRLINE']:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))

            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.write(f"**Prediction:** {'🟥 Delayed' if prediction == 1 else '🟩 On Time'}")
            st.write(f"**Delay Probability:** {prob:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.error("❌ Failed to load data.")
