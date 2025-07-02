from flask import Flask, request, render_template
import joblib
import pandas as pd
#changing file to see commit name

app = Flask(__name__)

# Load the trained model
model = joblib.load("knn.joblib")

# Load encoders
le_region = joblib.load("le_Region.joblib")
le_soil = joblib.load("le_Soil_Type.joblib")
le_crop = joblib.load("le_Crop.joblib")
le_weather = joblib.load("le_Weather_Condition.joblib")

# Load MinMaxScaler
scaler = joblib.load("minmax_scaler.joblib")

# Preprocessing function
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Apply label encoders
    df['Region'] = le_region.transform(df['Region'])
    df['Soil_Type'] = le_soil.transform(df['Soil_Type'])
    df['Crop'] = le_crop.transform(df['Crop'])
    df['Weather_Condition'] = le_weather.transform(df['Weather_Condition'])

    # Map binary values
    df['Irrigation_Used'] = 1 if df['Irrigation_Used'].iloc[0] == 'Yes' else 0
    df['Fertilizer_Used'] = 1 if df['Fertilizer_Used'].iloc[0] == 'Yes' else 0

    # Scale Rainfall and Temperature (leaving target Yield as dummy 0)
    scaled_values = scaler.transform([
        [df['Rainfall_mm'].iloc[0], df['Temperature_Celsius'].iloc[0], 0]
    ])[0]

    df['Rainfall_mm'] = scaled_values[0]
    df['Temperature_Celsius'] = scaled_values[1]

    # Return features in correct order
    return df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm',
               'Temperature_Celsius', 'Fertilizer_Used',
               'Irrigation_Used', 'Weather_Condition']]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        user_input = {
            'Region': request.form['Region'],
            'Soil_Type': request.form['Soil_Type'],
            'Crop': request.form['Crop'],
            'Rainfall_mm': float(request.form['Rainfall_mm']),
            'Temperature_Celsius': float(request.form['Temperature_Celsius']),
            'Fertilizer_Used': request.form['Fertilizer_Used'],
            'Irrigation_Used': request.form['Irrigation_Used'],
            'Weather_Condition': request.form['Weather_Condition']
        }

        # Preprocess and predict
        X = preprocess_input(user_input)
        scaled_pred = model.predict(X)[0]
        final_yield = scaler.inverse_transform([[0, 0, scaled_pred]])[0][2]

        return render_template('index.html', prediction=round(final_yield, 2))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)