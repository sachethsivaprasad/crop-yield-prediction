# Crop Yield Prediction using Machine Learning

This project predicts crop yield (in tons per hectare) based on environmental and agricultural parameters using machine learning.

## Abstract

This project focuses on predicting agricultural crop yield based on key environmental and cultivation parameters using machine learning techniques. The workflow began with Exploratory Data Analysis (EDA) to understand the relationships among variables such as region, soil type, crop type, rainfall, temperature, fertiliser and irrigation usage, and weather conditions.

Subsequent data preprocessing involved encoding categorical features, scaling numerical values, and transforming binary inputs to prepare the dataset for modelling. We experimented with multiple regression models, including Linear Regression, K-Nearest Neighbours (KNN) Regression, etc. to identify the best performing algorithm for yield prediction.

After evaluating the models using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score, the final trained model was saved using joblib. A user-friendly web interface was developed using Python Flask, allowing users to input agricultural parameters and receive yield predictions. The complete application was successfully deployed to a cloud platform, making the predictive system accessible via a web browser.

## Technologies Used

* Python
* Pandas, NumPy
* scikit-learn
* Flask
* HTML/CSS

## Project Structure

```bash
crop-yield-prediction/
├── static/                # CSS/images
├── templates/             # HTML templates
├── main.py                # Flask app
├── preprocess.py          # Data transformation functions
├── requirements.txt
├── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sachethsivaprasad/crop-yield-prediction.git
cd crop-yield-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
python main.py
```

Visit `http://localhost:5000` in your browser.

## Input Parameters

The web app takes the following input fields:

* Region (North, South, East, West)
* Soil Type (Clay, Loam, Sandy, Peaty, Chalky, Silt)
* Crop (Rice, Maize, Wheat, Barley, Cotton, Soybean)
* Rainfall (mm)
* Temperature (Celsius)
* Fertiliser Used (Yes/No)
* Irrigation Used (Yes/No)
* Weather Condition (Sunny, Rainy, Cloudy)

## Acknowledgments

* Dataset: [Kaggle - Agriculture Crop Yield Dataset](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield)
* Model Hosting: [Hugging Face](https://huggingface.co/skcept/crop-yield-prediction)
* Website Hosting: [Rener](https://crop-yield-prediction-063b.onrender.com)
