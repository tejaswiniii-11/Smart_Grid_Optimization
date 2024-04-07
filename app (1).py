from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
from io import BytesIO
import os
from datetime import datetime
import threading

app = Flask(__name__)

# Load the saved XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Shared variable to store the plot
plot_data = None
plot_lock = threading.Lock()

def generate_plot(start_date, end_date, X_test, predictions):
    global plot_data, plot_lock

    # Generate a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.date_range(start=start_date, end=end_date, freq='D'), predictions)
    plt.title('Predicted Wind Power Generation')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')

    # Save the plot as a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    with plot_lock:
        plot_data = img_bytes

@app.route('/', methods=['GET', 'POST'])
def predict_power():
    global plot_data, plot_lock

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Convert input dates to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Generate dates for the requested time period
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create a DataFrame to hold the data
        requested_data = pd.DataFrame({'DateTime': date_range})

        # Assuming you have features similar to your original dataset
        # Replace these with the actual features you would expect for those months
        feature_names = ['Air temperature | (°C)', 'Pressure | (atm)', 'Wind speed | (m/s)']
        for feature_name in feature_names:
            requested_data[feature_name] = np.random.randn(len(date_range))

        # Prepare features for prediction
        X_test = requested_data[feature_names]

        # Make predictions for the requested time period
        predictions = loaded_model.predict(X_test)

        # Calculate the total power generated
        total_power_generated = predictions.sum()

        # Generate the plot in a background thread
        plot_thread = threading.Thread(target=generate_plot, args=(start_date, end_date, X_test, predictions))
        plot_thread.start()

        return render_template('index.html', predicted_power=predictions.tolist(),
                               start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
                               total_power_generated=total_power_generated)

    # Reset the plot data
    with plot_lock:
        plot_data = None

    return render_template('index.html')

@app.route('/download')
def download_predictions():
    # Get the start date, end date, and predicted power values from the request
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    predicted_power = request.args.get('predicted_power')

    # Create a DataFrame to hold the predicted power
    date_range = pd.date_range(start=datetime.strptime(start_date, '%Y-%m-%d'),
                               end=datetime.strptime(end_date, '%Y-%m-%d'), freq='D')
    prediction_df = pd.DataFrame({'DateTime': date_range, 'Predicted_Power': predicted_power})

    # Save the predicted power to an Excel file
    prediction_df.to_excel('predicted_power.xlsx', index=False)
    return send_file('predicted_power.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)