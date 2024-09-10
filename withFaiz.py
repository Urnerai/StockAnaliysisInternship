import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import os
import random

# Constants
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(29)
np.random.seed(32)
tf.random.set_seed(20)

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used.")
else:
    print("GPU is not available. The code will run on CPU.")

# Data download
symbol = "MPARK.IS"
start_date = datetime(2022, 1, 26)
end_date = datetime(2023, 6, 1)
data = yf.download(tickers=symbol, start=start_date, end=end_date, interval="1d")
data = data.dropna()

# Feature selection and scaling
features = data[['Close']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

# Create dataset
def create_dataset(features, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(features.iloc[i + time_steps]['Close'])
    return np.array(X), np.array(y)

time_steps = 4
X, y = create_dataset(pd.DataFrame(scaled_features, columns=features.columns), time_steps)

# Model creation
def create_model():
    model = tf.keras.Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(256,  return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Check for NaN or infinite values
print(f"NaNs in X: {np.isnan(X).sum()}, Infs in X: {np.isinf(X).sum()}")
print(f"NaNs in y: {np.isnan(y).sum()}, Infs in y: {np.isinf(y).sum()}")

# Train model
model = create_model()
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
history = model.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=32, verbose=1)

# Save and load model
model.save('stock_price_model.h5')
loaded_model = load_model('stock_price_model.h5')

# Make predictions
predictions = loaded_model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_features.shape[1] - 1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_features.shape[1] - 1))), axis=1))[:, 0]

# Evaluation
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Plot real and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data.index[:], data['Close'], label='Gerçek')  # Plot actual 'Close' prices

plt.plot(data.index[-len(predictions):], predictions, label='Tahmin')
plt.xticks(rotation=90)
plt.title('Model Eğitimi')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True) # Limit to 10 ticks
plt.show()

# OLS Regression
X_train_summarized = np.mean(X_train, axis=1)
X_train_sm = sm.add_constant(X_train_summarized)
model_sm = sm.OLS(y_train, X_train_sm)
result = model_sm.fit()
print(result.summary())

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
