import pandas as pd
import numpy as np
import joblib

# ✅ Load the trained model & scaler
model = joblib.load("house_rent_model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Get expected feature names (from scaler)
expected_columns = list(scaler.feature_names_in_)

# ✅ User Input
size = float(input("Enter Size (sq ft): "))
bhk = int(input("Enter BHK: "))
bathroom = int(input("Enter Number of Bathrooms: "))
floor = int(input("Enter Floor (e.g., '5' for 5th floor, '0' for Ground): "))
furnishing_status = input("Enter Furnishing Status (Furnished/Semi-Furnished/Unfurnished): ").strip()
area_type = input("Enter Area Type (Super Area/Carpet Area/Built Area): ").strip()

# ✅ Create input DataFrame
input_data = pd.DataFrame({
    'Size': [size],
    'BHK': [bhk],
    'Bathroom': [bathroom],
    'Floor': [floor],
    'Furnishing Status_Furnished': [1 if furnishing_status == "Furnished" else 0],
    'Furnishing Status_Semi-Furnished': [1 if furnishing_status == "Semi-Furnished" else 0],
    'Furnishing Status_Unfurnished': [1 if furnishing_status == "Unfurnished" else 0],
    'Area Type_Carpet Area': [1 if area_type == "Carpet Area" else 0],
    'Area Type_Super Area': [1 if area_type == "Super Area" else 0],
    'Area Type_Built Area': [1 if area_type == "Built Area" else 0]
})

# ✅ Print feature alignment debug info
print("\n🔍 Expected features:", expected_columns)
print("🔍 Features passed to model:", list(input_data.columns))

# ✅ Ensure correct feature alignment
input_data = input_data.reindex(columns=expected_columns, fill_value=0)  # Match model's features

# ✅ Scale the input
input_data_scaled = scaler.transform(input_data)

# ✅ Predict the rent
predicted_rent = model.predict(input_data_scaled)[0]

# ✅ Display the result
print(f"\n🏠 Estimated Rent: ₹{predicted_rent:.2f}")
