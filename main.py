import numpy as np
import yfinance as yf
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Model dosya yolunu belirtin
model_path = 'stock_price_model.h5'

# Modeli yükleyin
model = tf.keras.models.load_model(model_path)

# Hisse senedi sembolünü belirtin
ticker = 'MPARK.IS'

# Veriyi Yahoo Finance'den indirin (Geniş zaman aralığı)
data = yf.download(ticker, start='2024-01-01', end='2024-09-01')

features = data[['Close']]

# Veriyi ölçeklendirin
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

# Pencere uzunluğunu belirleyin
time_steps = 4  # Eğitim kodunda kullanılan pencere uzunluğu ile aynı

# Girdi verisini oluşturun
x_input = []
for i in range(len(scaled_features) - time_steps):
    x_input.append(scaled_features[i:i + time_steps])
x_input = np.array(x_input)

# Model tahminlerini yapın
predictions = model.predict(x_input)

# Tahminleri uygun bir formata dönüştürün
predictions = scaler.inverse_transform(predictions)

# Gerçek ve tahmin verilerini hizalayın
data_predicted = data.iloc[time_steps:].copy()
data_predicted['Predicted'] = predictions

# Günlük değişim farkını hesaplayın
data_predicted['Change'] = data_predicted['Close'] - data_predicted['Predicted']

# Gelecek 30 gün için tahmin
last_sequence = scaled_features[-time_steps:].reshape((1, time_steps, 1))
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)

future_predictions = []
for _ in range(30):
    future_pred = model.predict(last_sequence)
    future_predictions.append(future_pred[0, 0])
    new_input = np.concatenate((last_sequence[:, 1:, :], future_pred.reshape((1, 1, 1))), axis=1)
    last_sequence = new_input

# Gelecek tahminlerini bir DataFrame'e ekleyin
future_predictions_Array = np.array(future_predictions).reshape(-1, 1)
future_predictions_Array = scaler.inverse_transform(future_predictions_Array.reshape(-1, 1))
future_df = pd.DataFrame(future_predictions_Array, index=future_dates, columns=['Predicted'])
print("Future predictions (first 5):", future_predictions_Array[:5])

# Yalnızca son iki ayı çizdirme
filtered_data = data.loc['2024-07-01':'2024-09-01']
filtered_data_predicted = data_predicted.loc['2024-07-01':'2024-09-01']

# Sonuçları çizdirme (sadece son iki ay)
fig, axs = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

# İlk grafik: Fiyat verisi (Son iki ay)
axs[0].plot(filtered_data.index, filtered_data['Close'], label='Gerçek', color='blue')
axs[0].plot(filtered_data_predicted.index, filtered_data_predicted['Predicted'], label='Tahmin (Geçmiş)', color='red')
axs[0].plot(future_df.index, future_df['Predicted'], label='Gelecek 30 Gün Tahmini', color='green', linestyle='--')
axs[0].set_ylabel('Kapanış Fiyatı (TL)')
axs[0].legend(loc='upper left')
axs[0].grid(True)
axs[0].set_title(f'{ticker} Fiyat Tahmini (Son 2 Ay)')

# İkinci grafik: Günlük değişim farkı (Son iki ay)
axs[1].plot(filtered_data_predicted.index, filtered_data_predicted['Change'], label='Günlük Değişim Farkı', color='purple', linestyle='--')
axs[1].set_xlabel('Tarih')
axs[1].set_ylabel('Fark (TL)')
axs[1].legend(loc='upper left')
axs[1].grid(True)
axs[1].set_title('Günlük Değişim Farkı (Son 2 Ay)')

plt.tight_layout()
plt.show()
