import pandas as pd
import numpy as np
import random
import os

# Number of patients per file
num_patients = 100  

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Directory to save datasets
output_dir = "synthetic_datasets"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def random_age():
    return np.random.randint(18, 90)

def random_sex():
    return random.choice(["Male", "Female"])

def random_comorbidities():
    return random.choice([0, 1, 2, 3])

def random_crp():
    return round(np.random.uniform(0, 200), 1)

def random_wbc():
    return round(np.random.uniform(4, 20), 1)

def random_fever():
    return round(np.random.uniform(36, 41), 1)

def oxygen_saturation_from_crp(crp):
    base = np.random.uniform(95, 100)
    drop = crp * np.random.uniform(0.05, 0.15)
    return round(max(70, base - drop), 1)

# -----------------------------
# Generate dataset based on severity distribution
# -----------------------------
def generate_dataset(severity_weights, filename):
    data = []
    for _ in range(num_patients):
        age = random_age()
        sex = random_sex()
        comorb = random_comorbidities()
        crp = random_crp()
        wbc = random_wbc()
        fever = random_fever()
        o2 = oxygen_saturation_from_crp(crp)

        # Generate severity based on score
        score = 0
        score += 1 if age > 60 else 0
        score += comorb
        score += 1 if crp > 50 else 0
        score += 1 if fever > 38 else 0
        score += 1 if o2 < 92 else 0

        # Determine severity probabilistically using severity_weights
        # severity_weights = {"Mild":0.7, "Moderate":0.2, "Severe":0.1} etc
        severities = ["Mild", "Moderate", "Severe"]
        severity_probs = [severity_weights[s] for s in severities]
        severity = np.random.choice(severities, p=severity_probs)

        # ICU Risk based on severity
        icu_prob = 0.05
        if severity == "Moderate":
            icu_prob = 0.3
        elif severity == "Severe":
            icu_prob = 0.7
        icu_risk = 1 if np.random.rand() < icu_prob else 0

        data.append({
            "Age": age,
            "Sex": sex,
            "Comorbidities": comorb,
            "CRP": crp,
            "WBC": wbc,
            "Fever": fever,
            "Oxygen_Saturation": o2,
            "Severity": severity,
            "ICU_Risk": icu_risk
        })

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"✅ Generated '{filename}' with distribution: {severity_weights}")

# -----------------------------
# Generate multiple scenarios
# -----------------------------
# 1. Balanced dataset
generate_dataset({"Mild": 0.33, "Moderate": 0.34, "Severe": 0.33}, "balanced_dataset.csv")

# 2. Mild-heavy dataset
generate_dataset({"Mild": 0.7, "Moderate": 0.2, "Severe": 0.1}, "mild_heavy_dataset.csv")

# 3. Severe-heavy dataset
generate_dataset({"Mild": 0.1, "Moderate": 0.2, "Severe": 0.7}, "severe_heavy_dataset.csv")
