import streamlit as st
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")

st.title("Flight Delay Predictor ✈️")
st.markdown("Enter flight details to estimate the probability of a delay:")

@st.cache_data
def load_data():
    try:
        # Adjust the path pattern as needed (e.g., if in subfolder use "data/Flight_data_*.csv")
        file_paths = sorted(glob.glob("Flight_data__part_*.csv"))
        df_list = [pd.read_csv(fp) for fp in file_paths]
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data
def preprocess_data(df):
    encoders = {}
    for col in ['Month', 'ORIGIN', 'DEST', 'AIRLINE']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

@st.cache_resource
def train_model():
    df = load_data()
    if df.empty:
        return None, None

    df, encoders = preprocess_data(df)
    X = df[['Month', 'ORIGIN', 'DEST', 'AIRLINE']]
    y = df['Delayed']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, encoders

model, encoders = train_model()

if model is None:
    st.error("❌ Could not train the model.")
else:
    st.success("✅ Model trained successfully!")

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

            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.write(f"**Prediction:** {'🟥 Delayed' if prediction == 1 else '🟩 On Time'}")
            st.write(f"**Delay Probability:** {prob:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
