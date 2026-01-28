import joblib
import numpy as np

# Load saved objects
model = joblib.load('models/severity_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')

# Sample input: Age, Blood_Pressure, Heart_Rate, WBC_Count, Symptom_Score
sample_input = np.array([[50, 140, 90, 11000, 6]])

# Scale input
sample_input_scaled = scaler.transform(sample_input)

# Predict
pred = model.predict(sample_input_scaled)
pred_label = le.inverse_transform(pred)
print(f"Predicted Disease Severity: {pred_label[0]}")
