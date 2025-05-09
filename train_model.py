import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
files = [
    "Flight_data_part_1.csv", "Flight_data_part_2.csv",
    "Flight_data_part_3.csv", "Flight_data_part_4.csv",
    "Flight_data_part_5.csv", "Flight_data_part_6.csv"
]
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
df = df.dropna(subset=['Month', 'AIRLINE', 'ORIGIN', 'DEST', 'Delayed'])

# Label encoding
le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['AIRLINE_ENC'] = le_airline.fit_transform(df['AIRLINE'])
df['ORIGIN_ENC'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_ENC'] = le_dest.fit_transform(df['DEST'])

# Train model
X = df[['Month', 'AIRLINE_ENC', 'ORIGIN_ENC', 'DEST_ENC']]
y = df['Delayed'].astype(int)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "delay_model.pkl")
joblib.dump(le_airline, "le_airline.pkl")
joblib.dump(le_origin, "le_origin.pkl")
joblib.dump(le_dest, "le_dest.pkl")
print("✅ Model and encoders saved.")
