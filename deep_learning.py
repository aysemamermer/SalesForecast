import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import model_selection
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

#%%

df = pd.read_excel('new_filtered.xlsx')

# df['Tarih'] = pd.to_datetime(df['Tarih'])
#%%

df.info()
df = df.dropna()
print(df.head())
print(df.columns)

df_month = df.copy()

df_month['Tarih'] = pd.to_datetime(df_month['Tarih'])

# Tarihlere göre sıralama yapın
df_month = df_month.sort_values('Tarih')

# Hafta numaralarını hesaplayın
df_month['Ay'] = (df_month['Tarih'].dt.year - df_month['Tarih'].dt.year.min()) * 12 + df_month['Tarih'].dt.month

df_month = df_month.drop(['Yıl', 'Müşteri Kodu', 'Toplam', 'Net Tutar'], axis = 1)

print(df_month.head())

#%%
melt = df_month.copy()

melt['Product_Code'] = df_month['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt['Month'] = df_month['Ay']
melt['Sales'] = df_month['Miktar']
melt['Price'] = df_month['Birim Fiyat']
melt['Date'] = df_month['Tarih']

melt = melt.drop(['Ürün Kodu','Ay', 'Birim Fiyat', 'Tarih', 'Miktar'], axis = 1)

print(melt)

dataset = melt.pivot_table(index = ['Product_Code'],values = ['Sales'],columns = ['Month'],fill_value = 0,aggfunc='sum')
print(dataset.head(3))



melt = melt.sort_values(['Month', 'Product_Code'])
product_num = melt['Product_Code'].nunique()
print("Farklı Ürün Sayısı", product_num)

print(melt.info())
label_encoder = LabelEncoder()
melt['Product_Code'] = label_encoder.fit_transform(melt['Product_Code']) + 1  # +1 ekleyerek 0'dan değil, 1'den başlatın
print(melt.head())


pro_selected = melt.groupby("Product_Code")["Sales"].sum().sort_values(ascending=False).index[0]
df_pro_selected = melt[melt["Product_Code"] == pro_selected]


df_pro_selected= df_pro_selected.sort_values(by=['Product_Code'], ascending=True)
print(df_pro_selected.info())


#%%

from sklearn.metrics import mean_squared_error
from math import sqrt

melt['Year'] = melt['Date'].dt.year
melt = melt.drop(['Date'], axis=1)

#%%

print(melt)
melt['Product_Code'] = melt['Product_Code'].astype('category').cat.codes

# Veriyi ölçeklendirin (0 ile 1 arasında)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(melt[['Sales', 'Price']])  # 'Product_Code' ölçeklendirmeye gerek yok

# LSTM modeli için girdi ve çıktıyı oluşturun
features = ['Product_Code', 'Month', 'Sales', 'Price']
X = scaled_data
y = scaled_data[:, 0]  # 'Sales' sütunu

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Girdi verilerini uygun şekilde yeniden şekillendirin
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

#%%
# LSTM modelini oluşturun
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitin
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Modelin performansını değerlendirin
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE:', rmse)



