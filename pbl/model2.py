



import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Step 1: Load the dataset
file_path = r"C:\Users\patne\pbl\data.csv"  # Update if needed

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
    print("🔍 Columns in dataset:", df.columns)

except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
    exit()

# ✅ Step 2: Display dataset info
print(df.head())
print(df.info())
print(df.describe())

# ✅ Step 3: Handle missing values
df = df.dropna()

# ✅ Step 4: Fix "Floor" Column (Convert Text to Number)
def extract_floor(floor_value):
    if "Ground" in str(floor_value):
        return 0  # Ground floor
    try:
        return int(str(floor_value).split()[0])  # Extract first number
    except ValueError:
        return np.nan  # Handle unexpected values

df["Floor"] = df["Floor"].apply(extract_floor)
df.dropna(subset=["Floor"], inplace=True)  # Drop rows with missing Floor values

# ✅ Step 5: Define Features (X) and Target (y)
X = df[['Size', 'BHK', 'Bathroom', 'Floor', 'Furnishing Status', 'Area Type']]  
y = df['Rent']

# ✅ Step 6: Convert Categorical Columns to Numeric (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# ✅ Print feature names (IMPORTANT for matching in predict.py)
print("\n🔍 Features used in training:", list(X.columns))

# ✅ Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 8: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Step 9: Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ✅ Step 10: Make Predictions
y_pred = model.predict(X_test_scaled)

# ✅ Step 11: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔍 Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ✅ Step 12: Save Model & Scaler
joblib.dump(model, "house_rent_model.pkl")
print("Features used during training:", list(X.columns))

joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
































# Most correct 2 model.py



# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # ✅ Step 1: Load the dataset
# file_path = r"C:\Users\patne\pbl\data.csv"  # Change if needed

# try:
#     df = pd.read_csv(file_path)
#     print("✅ Dataset loaded successfully!")
#     print("🔍 Columns:", df.columns)
# except FileNotFoundError:
#     print(f"❌ File not found at {file_path}")
#     exit()

# # ✅ Step 2: Data cleaning
# df.dropna(inplace=True)

# # ✅ Step 3: Fix Floor Column
# def extract_floor(floor_value):
#     if "Ground" in str(floor_value):
#         return 0
#     try:
#         return int(str(floor_value).split()[0])
#     except ValueError:
#         return np.nan

# df["Floor"] = df["Floor"].apply(extract_floor)
# df.dropna(subset=["Floor"], inplace=True)

# # ✅ Step 4: Drop outliers in Rent
# df = df[df["Rent"] < 200000]  # Remove extreme values

# # ✅ Step 5: Features and target
# X = df[['Size', 'BHK', 'Bathroom', 'Floor', 'Furnishing Status', 'Area Type', 'City']]
# y = np.log1p(df['Rent'])  # Log-transform Rent

# # ✅ Step 6: One-hot encoding
# X = pd.get_dummies(X, drop_first=True)

# # ✅ Step 7: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Step 8: Feature scaling using RobustScaler
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ✅ Step 9: Train model
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # ✅ Step 10: Predict and evaluate
# y_pred = model.predict(X_test_scaled)

# mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
# mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
# r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

# print("\n🔍 Evaluation:")
# print(f"MAE: ₹{mae:.2f}")
# print(f"MSE: ₹{mse:.2f}")
# print(f"R² Score: {r2:.2f}")

# # ✅ Step 11: Save model and scaler
# joblib.dump(model, "house_rent_model.pkl")
# joblib.dump(scaler, "scaler.pkl")

# print("✅ Model and scaler saved successfully!")
# print("📦 Features used:", list(X.columns))




# import  pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # ✅ Step 1: Load the dataset
# file_path = r"C:\Users\patne\pbl\data.csv"  # Update this if needed

# try:
#     df = pd.read_csv(file_path)
#     print("✅ Dataset loaded successfully!")
# except FileNotFoundError:
#     print(f"❌ File not found at {file_path}")
#     exit()

# # ✅ Optional: Preview unique localities (for debugging feature mismatch later)
# print("\n📍 Unique Area Localities:")
# print(df["Area Locality"].unique())

# # ✅ Step 2: Clean data
# df.dropna(inplace=True)

# # ✅ Step 3: Fix Floor column
# def extract_floor(floor_value):
#     if "Ground" in str(floor_value):
#         return 0
#     try:
#         return int(str(floor_value).split()[0])
#     except ValueError:
#         return np.nan

# df["Floor"] = df["Floor"].apply(extract_floor)
# df.dropna(subset=["Floor"], inplace=True)

# # ✅ Step 4: Remove outlier rents
# df = df[df["Rent"] < 200000]

# # ✅ Step 5: Define features and target
# X = df[['Size', 'BHK', 'Bathroom', 'Floor', 'Furnishing Status', 'Area Locality', 'City']]
# y = np.log1p(df['Rent'])  # log(Rent + 1)

# # ✅ Step 6: One-hot encoding
# X = pd.get_dummies(X, drop_first=True)

# # ✅ Step 7: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Step 8: Scaling
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ✅ Step 9: Train model
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # ✅ Step 10: Evaluate model
# y_pred = model.predict(X_test_scaled)

# mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
# mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
# r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

# print("\n📊 Evaluation Metrics:")
# print(f"MAE: ₹{mae:.2f}")
# print(f"MSE: ₹{mse:.2f}")
# print(f"R² Score: {r2:.2f}")

# # ✅ Step 11: Save model, scaler, and features
# joblib.dump(model, "house_rent_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(X.columns.tolist(), "feature_names.pkl")

# print("\n✅ Model, scaler, and feature names saved successfully.")
