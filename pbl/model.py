import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Load the dataset
file_path = "data.csv"  # Update if necessary
df = pd.read_csv(file_path)

# Display initial data summary
print("Initial dataset shape:", df.shape)
print("Checking for missing values:\n", df.isnull().sum())

# Data Preprocessing
df = df.dropna()  # Remove rows with missing values
print("Dataset shape after removing NaN rows:", df.shape)

# Feature Engineering: Creating Total_Rooms
df["Total_Rooms"] = df["BHK"] + df["Bathroom"]

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)
print("Shape after one-hot encoding:", df.shape)

# Define target variable and features
y = df["Rent"]  
X = df[["Total_Rooms", "Size"]]  # Consider Total Rooms instead of Bathroom

# Remove outliers using IQR method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

outlier_mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[outlier_mask]
y = y.loc[X.index]

print("Dataset size after outlier removal:", X.shape)

# Check for missing/infinite values
print("Missing values before VIF calculation:\n", X.isna().sum().sum())
print("Checking for infinite values in X:", np.isinf(X).sum().sum())

# Ensure data is numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1)
print("Shape after cleaning:", X.shape)

# VIF Calculation
if X.shape[1] > 1:  # Only calculate VIF if more than 1 feature exists
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("Initial VIF Scores:\n", vif_data)

    # Drop highly collinear feature
    while vif_data["VIF"].max() > 10:
        drop_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"Dropping feature due to high VIF: {drop_feature}")
        X = X.drop(columns=[drop_feature])

        if X.shape[1] <= 1:
            print("Only one feature left, stopping VIF calculation.")
            break

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("Updated VIF Scores:\n", vif_data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Train a regression model
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Print model summary
print(model.summary())

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Predict rent for test set
import statsmodels.api as sm
X_test = sm.add_constant(X_test)  # Add intercept term
y_pred = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot predictions vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Predicted vs Actual Rent")
plt.show()

import joblib

# Save the trained model
joblib.dump(model, "house_rent_model.pkl")
print("Model saved successfully as house_rent_model.pkl")






















#most correct

