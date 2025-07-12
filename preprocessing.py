import pandas as pd

def preprocess_input(data_dict, model_loader):
    df = pd.DataFrame([data_dict])
    # Apply label encoders
    df['Region'] = model_loader.le_region.transform(df['Region'])
    df['Soil_Type'] = model_loader.le_soil.transform(df['Soil_Type'])
    df['Crop'] = model_loader.le_crop.transform(df['Crop'])
    df['Weather_Condition'] = model_loader.le_weather.transform(df['Weather_Condition'])
    # Map binary values
    df['Irrigation_Used'] = 1 if df['Irrigation_Used'].iloc[0] == 'Yes' else 0
    df['Fertilizer_Used'] = 1 if df['Fertilizer_Used'].iloc[0] == 'Yes' else 0
    # Scale Rainfall and Temperature (leaving target Yield as dummy 0)
    scaled_values = model_loader.scaler.transform([
        [df['Rainfall_mm'].iloc[0], df['Temperature_Celsius'].iloc[0], 0]
    ])[0]
    df['Rainfall_mm'] = scaled_values[0]
    df['Temperature_Celsius'] = scaled_values[1]
    # Return features
    return df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm',
               'Temperature_Celsius', 'Fertilizer_Used',
               'Irrigation_Used', 'Weather_Condition']] 