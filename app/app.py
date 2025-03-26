from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Load the dataset and train the model
df = pd.read_csv('data/owid-covid-data.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Drop rows where 'new_cases' or 'days_since_start' is NaN
df = df.dropna(subset=['new_cases', 'days_since_start'])

# Prepare data for training
X = df[['days_since_start']]
y = df['new_cases']

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    # Placeholder for visualizations (update paths or generate images as needed)
    visualizations = {
        'infection_rates': 'visualizations/infection_rates.png',
        'vaccination_trends': 'visualizations/vaccination_trends.png',
        'recovery_heatmap': 'visualizations/recovery_heatmap.png',
        'case_prediction': 'visualizations/case_prediction.png'
    }
    return render_template('index.html', visualizations=visualizations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the number of days from the form
        days = int(request.form['days'])
        
        # Make a prediction
        prediction = model.predict(np.array([[days]]))
        
        # Return the prediction as a response
        return f'Predicted cases after {days} days: {prediction[0]:.2f}'
    except ValueError:
        return "Invalid input. Please enter a valid number of days."

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production