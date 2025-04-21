MOST CORRECT 
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("house_rent_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names based on training data
feature_names = ['Size', 'BHK', 'Bathroom', 'Floor', 
                'Furnishing Status_Semi-Furnished', 
                'Furnishing Status_Unfurnished', 
                'Area Type_Carpet Area', 
                'Area Type_Super Area']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        size = float(request.form['size'])
        bhk = int(request.form['bhk'])
        bathroom = int(request.form['bathroom'])
        floor = int(request.form['floor'])
        furnishing_status = request.form['furnishing']
        area_type = request.form['area']

        # Create an input dictionary
        input_dict = {
            'Size': [size],
            'BHK': [bhk],
            'Bathroom': [bathroom],
            'Floor': [floor],
            'Furnishing Status_Semi-Furnished': [1 if furnishing_status == 'Semi-Furnished' else 0],
            'Furnishing Status_Unfurnished': [1 if furnishing_status == 'Unfurnished' else 0],
            'Area Type_Carpet Area': [1 if area_type == 'Carpet Area' else 0],
            'Area Type_Super Area': [1 if area_type == 'Super Area' else 0]
        }

        # Convert to DataFrame
        input_data = pd.DataFrame(input_dict)

        # Scale the input
        input_data_scaled = scaler.transform(input_data)

        # Predict rent
        predicted_rent = model.predict(input_data_scaled)[0]

        return render_template("index.html", prediction=f"₹{predicted_rent:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)




























































# most correct 2



# from flask import Flask, render_template, request
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load("house_rent_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Features used during training (ensure the order matches what you printed in model.py)
# model_features = [
#     'Size', 'BHK', 'Bathroom', 'Floor',
#     'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished',
#     'Area Type_Carpet Area', 'Area Type_Super Area',
#     'City_Chennai', 'City_Delhi', 'City_Hyderabad',
#     'City_Kolkata', 'City_Mumbai'
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         size = float(request.form['size'])
#         bhk = int(request.form['bhk'])
#         bathroom = int(request.form['bathroom'])
#         floor = int(request.form['floor'])
#         furnishing = request.form['furnishing']
#         area = request.form['area']
#         city = request.form['city']

#         # Base input
#         input_data = {
#             'Size': size,
#             'BHK': bhk,
#             'Bathroom': bathroom,
#             'Floor': floor,
#             'Furnishing Status_Semi-Furnished': 0,
#             'Furnishing Status_Unfurnished': 0,
#             'Area Type_Carpet Area': 0,
#             'Area Type_Super Area': 0,
#             'City_Chennai': 0,
#             'City_Delhi': 0,
#             'City_Hyderabad': 0,
#             'City_Kolkata': 0,
#             'City_Mumbai': 0
#         }

#         # Handle categorical encoding manually
#         if furnishing == "Semi-Furnished":
#             input_data['Furnishing Status_Semi-Furnished'] = 1
#         elif furnishing == "Unfurnished":
#             input_data['Furnishing Status_Unfurnished'] = 1

#         if area == "Carpet Area":
#             input_data['Area Type_Carpet Area'] = 1
#         elif area == "Super Area":
#             input_data['Area Type_Super Area'] = 1

#         if city in ["Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]:
#             input_data[f"City_{city}"] = 1  # Skip Bangalore (drop_first=True)

#         # Convert to DataFrame
#         df = pd.DataFrame([input_data])
#         df_scaled = scaler.transform(df)

#         prediction = model.predict(df_scaled)[0]
#         predicted_rent = f"₹{prediction:,.2f}"

#         return render_template('result.html', rent=prediction, city=city, furnishing=furnishing, size=size, bhk=bhk,
#                                bathroom=bathroom, floor=floor, area_type=area)

#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == '__main__':
#     app.run(debug=True)






































































































































































# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import numpy as np


# app = Flask(__name__)

# # Load model, scaler, and feature names
# model = joblib.load("house_rent_model.pkl")
# scaler = joblib.load("scaler.pkl")
# feature_names = joblib.load("feature_names.pkl")

# # Load original dataset to extract city-locality mapping
# df = pd.read_csv("data.csv")
# df.dropna(subset=["City", "Area Locality"], inplace=True)

# # Build mapping: City -> Area Localities
# city_locality_map = {}
# for city in df["City"].unique():
#     localities = sorted(df[df["City"] == city]["Area Locality"].unique().tolist())
#     city_locality_map[city] = localities

# @app.route("/")
# def home():
#     cities = list(city_locality_map.keys())
#     return render_template("index.html", cities=cities, city_locality_map=city_locality_map)

# @app.route("/predict", methods=["POST"])
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get form inputs
#         size = float(request.form["size"])
#         bhk = int(request.form["bhk"])
#         bathroom = int(request.form["bathroom"])
#         floor = int(request.form["floor"])
#         furnishing = request.form["furnishing"]
#         area_type = request.form["area"]
#         city = request.form["city"]
#         locality = request.form["locality"]

#         # Start with base features
#         input_data = {
#             "Size": size,
#             "BHK": bhk,
#             "Bathroom": bathroom,
#             "Floor": floor,
#         }

#         # Add dummies ONLY if they exist in the trained feature set
#         for col in feature_names:
#             if col == f"City_{city}":
#                 input_data[col] = 1
#             elif col == f"Area Locality_{locality}":
#                 input_data[col] = 1
#             elif col == f"Furnishing Status_{furnishing}":
#                 input_data[col] = 1

#         # Fill in all missing columns with 0
#         full_input = {col: 0 for col in feature_names}
#         full_input.update(input_data)

#         # Prepare input for prediction
#         df_input = pd.DataFrame([full_input])
#         df_scaled = scaler.transform(df_input)

#         # Predict
#         prediction = model.predict(df_scaled)[0]
#         predicted_rent = f"₹{round(np.expm1(prediction), 2):,.2f}"

#         return render_template("result.html", rent=predicted_rent, city=city, locality=locality,
#                                size=size, bhk=bhk, bathroom=bathroom, floor=floor, furnishing=furnishing)

#     except Exception as e:
#         return f"❌ Error: {str(e)}"


# if __name__ == "__main__":
#     app.run(debug=True)
