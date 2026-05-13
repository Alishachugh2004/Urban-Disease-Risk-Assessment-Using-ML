# Urban Disease Risk Assessment

AI-Based Urban Disease Spread Risk Assessment Using Environmental Data

## Project Overview

This project uses machine learning to predict disease spread risk levels in urban areas based on environmental factors such as temperature, humidity, air quality index (AQI), and population density.

## Project Structure

```
Alisha_Project/
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   └── dataset.csv         # Training dataset
├── models/
│   ├── model.pkl           # Trained RandomForest model
│   └── scaler.pkl          # Feature scaler
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis
├── reports/
│   └── report.txt          # Model training report
├── src/
│   ├── preprocess.py       # Data preprocessing module
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction module
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Features

- **Data Preprocessing**: Handle missing values, feature scaling, target variable creation
- **Machine Learning**: RandomForestClassifier for risk prediction
- **Web Dashboard**: Interactive Streamlit UI with:
  - Parameter sliders for environmental inputs
  - Real-time risk prediction
  - Dataset preview
  - Visualizations (charts, maps)
  - Alerts for high AQI and risk levels

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py
```

3. Run the Streamlit app:

```bash
streamlit run app/app.py
```

## Usage

1. Open the Streamlit app in your browser (typically http://localhost:8501)
2. Use the sidebar sliders to adjust:
   - Temperature (0-50°C)
   - Humidity (0-100%)
   - AQI (0-500)
   - Population Density (0-10000)
3. Click "Predict Risk" to get the risk assessment
4. View the prediction results, confidence scores, and any alerts

## Model Details

- **Algorithm**: RandomForestClassifier
- **Features**: Temperature, Humidity, AQI, Population Density
- **Target**: Risk Level (Low/Medium/High)
- **Train/Test Split**: 80/20

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## License

MIT License