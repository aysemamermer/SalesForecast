import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Veriyi okuyun
df = pd.read_excel('data_new.xlsx')
df['Tarih'] = pd.to_datetime(df['Tarih'], format='%Y-%m-%d')
df['Ürün Kodu'] = df['Ürün Kodu'].astype(str)
df['Miktar'] = df['Miktar'].astype('float')

# 'Ürün Kodu' sütununu kategorik olarak işaretleyin
label_encoder = LabelEncoder()
df['Ürün Kodu'] = label_encoder.fit_transform(df['Ürün Kodu'])

# 'Miktar' sütununu ölçeklendirin
scaler = StandardScaler()
df['Miktar_Scaled'] = scaler.fit_transform(df['Miktar'].values.reshape(-1, 1))

# Eğitim ve test verilerini ayırın
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Modeli tanımlayın
model = Sequential()
model.add(Embedding(input_dim=len(df['Ürün Kodu'].unique()), output_dim=128, input_length=1))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Modeli derleyin
model.compile(optimizer='adam', loss='mse')

# Eğitim verileri üzerinde modeli eğitin
model.fit(train_data['Ürün Kodu'], train_data['Miktar_Scaled'], epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# Test verileri üzerinde tahmin yapın
predictions_scaled = model.predict(test_data['Ürün Kodu'])

# Orijinal değerlere dönüştürme
predictions = scaler.inverse_transform(predictions_scaled)

# Modelin performansını değerlendirin
print(model.evaluate(test_data['Ürün Kodu'], test_data['Miktar_Scaled']))

# Tahmin edilecek tarih aralığını tanımlayın
predict_period_dates = pd.date_range(start='2022-07-01', end='2023-01-31', freq='D')

# Modeli kullanarak tahminler yapın
y_pred_future_scaled = model.predict(np.zeros((len(predict_period_dates), 1)))

# Orijinal değerlere dönüştürme
y_pred_future = scaler.inverse_transform(y_pred_future_scaled)

# Tahminleri veri çerçevesine ekleyin
df_forecast = pd.DataFrame({'Date': predict_period_dates, 'Satış Miktarı': y_pred_future.flatten()})

# Gerçek satış miktarlarını belirli bir tarih aralığına filtreleyin
start_date = '2022-07-01'
end_date = '2023-01-31'
filtered_original = df[(df['Tarih'] >= start_date) & (df['Tarih'] <= end_date)][['Tarih', 'Miktar']]

# Orijinal ve tahmin edilen satış miktarlarını içeren bir görselleştirme yapın
plt.figure(figsize=(10, 6))
plt.title('Aylık Satış Miktarı Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Miktar')
sns.lineplot(data=filtered_original, x='Tarih', y='Miktar', label='Gerçek Satış Miktarı')
sns.lineplot(data=df_forecast, x='Date', y='Satış Miktarı', label='Tahmin Edilen Satış Miktarı')
plt.legend()
plt.show()
