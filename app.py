import yfinance as yf
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import math

app = Flask(__name__)

# Load the saved LSTM model
model = load_model('stock_prediction_model.h5')

# Home route to display form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle stock prediction
@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock']
    data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

    if data.empty:
        return "Invalid stock symbol or no data available"

    # Use 'Close' prices for prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare the data (same logic as before)
    time_step = 60
    def create_sequences(data, time_step):
        sequences, labels = [], []
        for i in range(len(data) - time_step - 1):
            sequences.append(data[i:(i + time_step), 0])
            labels.append(data[i + time_step, 0])
        return np.array(sequences), np.array(labels)

    X_data, _ = create_sequences(scaled_data, time_step)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

    # Make predictions
    predictions = model.predict(X_data)
    predictions = scaler.inverse_transform(predictions)

    # Plot the results and save as image
    plt.figure(figsize=(14, 6))
    plt.plot(close_prices[time_step:], label="Actual Price", color='blue')
    plt.plot(predictions, label="Predicted Price", color='red')
    plt.title(f'Stock Price Prediction for {stock_symbol}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('static/stock_prediction.png')

    # Display the plot and RMSE on a result page
    return render_template('result.html', stock=stock_symbol)

if __name__ == '__main__':
    app.run(debug=True)
