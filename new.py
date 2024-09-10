import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import statsmodels.api as sm
import os
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(29)
np.random.seed(32)
tf.random.set_seed(20)
# Veri indirme
symbol = "KRDMD.IS"
start_date = datetime(2022, 1, 26)
end_date = datetime(2024, 9, 1)
startStr = start_date.strftime("%Y-%m-%d")
endStr = end_date.strftime("%Y-%m-%d")

data = yf.download(tickers=symbol, start=startStr, end=endStr, interval="1d")

# Eksik değerleri temizleme
data = data.dropna()

# Özelliklerin seçilmesi
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Özelliklerin ölçeklendirilmesi
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Dataset oluşturma fonksiyonu
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, 3])  # 'Close' fiyatı tahmini
    return np.array(X), np.array(y)

time_steps = 4
X, y = create_dataset(scaled_features, time_steps)

# Model oluşturma
model = Sequential()
model.add(LSTM(90, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(90))
model.add(Dropout(0.1))
model.add(Dense(1))

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0071245 ), loss='mean_squared_error')

# Modeli eğitme
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=1)

# Modeli kaydetme
model.save('stock_price_prediction_model.h5')

# Modeli yükleme
loaded_model = load_model('stock_price_prediction_model.h5')

# Tahmin yapma
predictions = loaded_model.predict(X_test)

# Tahminleri ölçekten geri çevirme
predictions = scaler.inverse_transform(
    np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_features.shape[1] - 1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_features.shape[1] - 1))), axis=1))[:, 0]

# Test doğruluğunu hesaplama
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Performans grafiği
plt.plot(y_test, label='Gerçek')
plt.plot(predictions, label='Tahmin')
plt.legend()
plt.show()

X_train_sm = sm.add_constant(X_train.reshape(X_train.shape[0], -1))  # Sabit terim ekleme
model_sm = sm.OLS(y_train, X_train_sm)
result = model_sm.fit()

# Sonuçları özetleme
print(result.summary())