import joblib

model = joblib.load("house_rent_model.pkl")
print("Expected number of input features:", model.model.exog.shape[1])
