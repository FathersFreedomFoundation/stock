import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
import os

data = pd.read_csv('stock_data.csv')

close_prices = data['Close'].values
close_prices = close_prices.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

training_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

def create_sequences(data, time_step):
    sequences = []
    labels = []
    for i in range(len(data) - time_step - 1):
        sequences.append(data[i:(i + time_step), 0])
        labels.append(data[i + time_step, 0])
    return np.array(sequences), np.array(labels)

time_step = 60
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=64, epochs=50)

model.save('stock_prediction_model.h5')

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform([y_test])

mse = mean_squared_error(y_test_actual[0], predictions)
rmse = math.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual[0], label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price", color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

plt.savefig('stock_price_prediction.png')
plt.show()

predictions_df = pd.DataFrame({'Actual': y_test_actual[0], 'Predicted': predictions.flatten()})
predictions_df.to_csv('predicted_vs_actual.csv', index=False)

print("Model, plot, and predictions have been saved.")
print(f"Current directory: {os.getcwd()}")
