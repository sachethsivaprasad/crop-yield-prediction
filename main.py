from flask import Flask, request, render_template, redirect, url_for, session
from model_loader import ModelLoader
from preprocessing import preprocess_input

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

model_loader = ModelLoader.get_instance()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/input', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
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
        X = preprocess_input(user_input, model_loader)
        scaled_pred = model_loader.model.predict(X)[0]
        final_yield = model_loader.scaler.inverse_transform([[0, 0, scaled_pred]])[0][2]
        session['prediction'] = round(final_yield, 2)
        session['input_data'] = user_input
        return redirect(url_for('result'))
    return render_template('input.html')

@app.route('/result')
def result():
    prediction = session.get('prediction', None)
    input_data = session.get('input_data', None)
    return render_template('result.html', prediction=prediction, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)