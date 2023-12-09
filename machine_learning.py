import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import model_selection
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

#%%
# Veri setini oku
df = pd.read_excel('new_filtered.xlsx')

df['Tarih'] = pd.to_datetime(df['Tarih'])

df.info()
df = df.dropna()
print(df.head())
print(df.columns)


#%% Verileri Aylık Olarak gruplayıp incele

parameters = ['Tarih', 'Yıl', 'Müşteri Kodu', 'Ürün Kodu', 'Miktar', 'Birim Fiyat',
       'Toplam', 'Net Tutar']


# Aylık olarak gruplama ve satışları toplama
monthly_sales = df.resample('M', on='Tarih')['Miktar'].sum().reset_index()

print(df.info())
# Aylık toplam satışları görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Tarih'], monthly_sales['Miktar'], marker='o')
plt.title('Aylık Toplam Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış (Miktar)')
plt.grid(True)
plt.show()

print(df.info())

#%%

toplam_satis_miktari = df['Miktar'].sum()
toplam_gelir = df['Toplam'].sum()

print(f'Toplam Satış Miktarı: {toplam_satis_miktari}')
print(f'Toplam Gelir: {toplam_gelir}')

# Ürün bazında satışları inceleyin
urun_bazinda_satis = df.groupby('Ürün Kodu')['Miktar'].sum().sort_values(ascending=False)
en_cok_satan_urun = urun_bazinda_satis.idxmax()
en_cok_satan_miktar = urun_bazinda_satis.max()

print(f'En Çok Satılan Ürün: {en_cok_satan_urun}')
print(f'En Çok Satılan Miktar: {en_cok_satan_miktar}')

# Müşteri bazında satın almaları değerlendirin
musteri_bazinda_satis = df.groupby('Müşteri Kodu')['Miktar'].sum().sort_values(ascending=False)
en_cok_satan_musteri = musteri_bazinda_satis.idxmax()
en_cok_satan_musteri_miktar = musteri_bazinda_satis.max()

print(f'En Çok Satın Alan Müşteri: {en_cok_satan_musteri}')
print(f'En Çok Satın Alma Miktarı: {en_cok_satan_musteri_miktar}')

# Görselleştirmeler
plt.figure(figsize=(12, 6))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Toplam Satış Miktarı ve Gelir
sns.barplot(x=['Toplam Satış Miktarı', 'Toplam Gelir'], y=[toplam_satis_miktari, toplam_gelir], ax=axes[0, 0])
axes[0, 0].set_title('Toplam Satış Miktarı ve Gelir')

# Ürün Bazında Satışlar
urun_bazinda_satis.head(15).plot(kind='bar', color='skyblue', ax=axes[0, 1])
axes[0, 1].set_title('Ürün Bazında Satışlar')

# Müşteri Bazında Satışlar
musteri_bazinda_satis.head(10).plot(kind='bar', color='lightcoral', ax=axes[1, 0])
axes[1, 0].set_title('Müşteri Bazında Satışlar')

# Boş subplot'u gizle
axes[1, 1].axis('off')

# Grafikleri düzenle
plt.tight_layout()
plt.show()

#%%
# Toplam Tutar ve Net Tutar Verilerini Görselleştirme
print(df.info())
#%%
df['Miktar'] = df['Miktar'].astype('float')
df['Birim Fiyat'] = df['Birim Fiyat'].astype('float')
df['Toplam'] = df['Toplam'].astype('float')
df['Net Tutar'] = df['Net Tutar'].astype('float')

df_new = df.copy()
df_copy = df.copy()
df_new['Tarih'] = pd.to_datetime(df_new['Tarih'])  # 'df_new' üzerinde işlem yapacağınızı unutmayın
df_new.set_index('Tarih', inplace=True)

# Aylık toplam tutar ve net tutarı al
monthly_total = df.resample('M', on='Tarih')['Toplam'].sum().reset_index()
monthly_net = df.resample('M', on='Tarih')['Net Tutar'].sum().reset_index()

# Görselleştirme
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
sns.lineplot(x='Tarih', y='Toplam', data=monthly_total, marker='o', color='blue')
plt.title('Aylık Toplam Tutar')

plt.subplot(1, 2, 2)
sns.lineplot(x='Tarih', y='Net Tutar', data=monthly_net, marker='o', color='green')
plt.title('Aylık Net Tutar')

plt.show()


#%%

# Ürün bazında Toplam Tutar ve Net Tutar Analizi
product_analysis = df.groupby('Ürün Kodu')[['Toplam', 'Net Tutar']].sum()
product_analysis = product_analysis.sort_values(by='Toplam', ascending=False)

# Görselleştirme
plt.figure(figsize=(12, 6))
product_analysis.head(10).plot(kind='bar', color=['blue', 'green'])
plt.title('Ürün Bazında Toplam Tutar ve Net Tutar Analizi')
plt.xlabel('Ürün Kodu')
plt.ylabel('Toplam Tutar ve Net Tutar')
plt.show()

# Müşteri bazında Toplam Tutar ve Net Tutar Analizi
customer_analysis = df.groupby('Müşteri Kodu')[['Toplam', 'Net Tutar']].sum()
customer_analysis = customer_analysis.sort_values(by='Toplam', ascending=False)

# Görselleştirme
plt.figure(figsize=(12, 6))
customer_analysis.head(20).plot(kind='bar', color=['blue', 'green'])
plt.title('Müşteri Bazında Toplam Tutar ve Net Tutar Analizi')
plt.xlabel('Müşteri Kodu')
plt.ylabel('Toplam Tutar ve Net Tutar')
plt.show()

#%%
print(df.info())
df = df.dropna()

print(df_new.info())

print(df_copy.info())

#%%  

en_cok_satis = df_copy.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün satış miktarını görselleştir
df_en_cok_satis = df_copy[df_copy["Ürün Kodu"] == en_cok_satis]

# Grafik boyutunu ve yazı tipini ayarlayın
plt.figure(figsize=(15, 5), dpi=120)
plt.rcParams["font.size"] = 14

# Grafik başlığını ve eksen etiketlerini ekleyin
plt.title(f"{en_cok_satis} Ürününün Satış Miktarı")
plt.xlabel("Tarih")
plt.ylabel("Miktar")

# Grafik renklerini ve çizgi stillerini seçin
plt.scatter(df_en_cok_satis["Tarih"], df_en_cok_satis["Miktar"], color="blue")

# Grafiklere açıklamalar ekleyin

plt.show()
#%%
# Ürün bazında miktarı gruplandır ve toplamı hesapla
product_quantity = df_copy.groupby("Ürün Kodu")["Miktar"].sum()

# Miktarı en çok olan ilk 10 ürünü sırala
product_quantity = product_quantity.sort_values(ascending=False).head(20)

# Görselleştirme
plt.figure(figsize=(12, 6))
product_quantity.plot(kind='bar', color="blue")
plt.title("Ürün Bazında Miktar Analizi")
plt.xlabel("Ürün Kodu")
plt.ylabel("Miktar")
plt.show()

#%%
df_copy_dropped = df_copy.copy()
df_copy_dropped = df_copy_dropped.dropna(subset=['Ürün Kodu'])

# Kontrol amaçlı yazdırma
print(df_copy_dropped.head())


df_filtered = df_copy_dropped[df_copy_dropped['Ürün Kodu'].str.startswith('Y') | df_copy_dropped['Ürün Kodu'].str.startswith('y')]

# Kontrol amaçlı yazdırma
print(df_filtered.head())

excel_file_path = "new_filtered.xlsx"
df_filtered.to_excel(excel_file_path, index=False)



#%% en çok satılan ürün

en_cok_satis = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün satış miktarını görselleştir
df_en_cok_satis = df_filtered[df_filtered["Ürün Kodu"] == en_cok_satis]

# Grafik boyutunu ve yazı tipini ayarlayın
plt.figure(figsize=(15, 5), dpi=120)
plt.rcParams["font.size"] = 14

# Grafik başlığını ve eksen etiketlerini ekleyin
plt.title(f"{en_cok_satis} Ürününün Satış Miktarı")
plt.xlabel("Tarih")
plt.ylabel("Miktar")

# Grafik renklerini ve çizgi stillerini seçin
plt.scatter(df_en_cok_satis["Tarih"], df_en_cok_satis["Miktar"], color="blue")

# Grafiklere açıklamalar ekleyin

plt.show()

#%% Filtered Aylık Toplam Satışlar

monthly_sales = df_filtered.resample('M', on='Tarih')['Miktar'].sum().reset_index()

print(df_filtered.info())
# Aylık toplam satışları görselleştirme
plt.figure(figsize=(15, 5))
plt.plot(monthly_sales['Tarih'], monthly_sales['Miktar'], marker='o')
plt.title('Aylık Toplam Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış (Miktar)')
plt.grid(True)
plt.show()

print(df_filtered.info())

#%%

# 'Ürün Kodu' sütunundaki farklı ürün tiplerini say
farkli_urun_tipleri = df_filtered['Ürün Kodu'].nunique()
# Sonucu yazdır
print(f"Filtered Farklı Ürün Tipleri Sayısı: {farkli_urun_tipleri}")

sayi_farkli_urun_tipleri = df_copy['Ürün Kodu'].nunique()
# Sonucu yazdır
print(f"Filtered Farklı Ürün Tipleri Sayısı: {sayi_farkli_urun_tipleri}")

#%% Miktarı en çok olan ilk 30 ürün

product_quantity = df_filtered.groupby("Ürün Kodu")["Miktar"].sum()

# Miktarı en çok olan ilk 30 ürünü sırala
product_quantity = product_quantity.sort_values(ascending=False).head(30)

# Görselleştirme
plt.figure(figsize=(12, 6))
product_quantity.plot(kind='bar', color="blue")
plt.title("Ürün Bazında Miktar Analizi")
plt.xlabel("Ürün Kodu")
plt.ylabel("Miktar")
plt.show()

#%%

en_cok_satis = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün satış miktarını görselleştir
df_en_cok_satis = df_filtered[df_filtered["Ürün Kodu"] == en_cok_satis]

# Grafik boyutunu ve yazı tipini ayarlayın
plt.figure(figsize=(15, 5), dpi=120)
plt.rcParams["font.size"] = 14
# Grafik başlığını ve eksen etiketlerini ekleyin
plt.title(f"{en_cok_satis} Ürününün Satış Miktarı")
plt.xlabel("Tarih")
plt.ylabel("Miktar")
# Grafik renklerini ve çizgi stillerini seçin
plt.scatter(df_en_cok_satis["Tarih"], df_en_cok_satis["Miktar"], color="blue")
plt.show()

#%% EN ÇOK SATILAN ÜRÜN AYLIK
en_cok_satis = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün satış miktarını aylık olarak grupla
df_en_cok_satis = df_filtered[df_filtered["Ürün Kodu"] == en_cok_satis].resample('M', on='Tarih')['Miktar'].sum().reset_index()

# Grafik boyutunu ve yazı tipini ayarla
plt.figure(figsize=(15, 5), dpi=120)
plt.rcParams["font.size"] = 14

# Çizgi grafiği çiz
plt.plot(df_en_cok_satis['Tarih'], df_en_cok_satis['Miktar'], marker='o', color="green", linestyle='-', linewidth=2)
plt.title(f"{en_cok_satis} Ürününün Aylık Satışları")
plt.xlabel('Tarih')
plt.ylabel('Satış Miktarı')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%%  haftalık toplam satışlar

weekly_sales = df_filtered.resample('W', on='Tarih')['Miktar'].sum().reset_index()

print(df_filtered.info())
# Haftalık toplam satışları görselleştirme
plt.figure(figsize=(15, 5))
plt.plot(weekly_sales['Tarih'], weekly_sales['Miktar'], marker='o')
plt.title('Haftalık Toplam Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış (Miktar)')
plt.grid(True)
plt.show()

print(df_filtered.info())

#%% Yıllık toplam satışlar

yearly_sales = df_filtered.resample('Y', on='Tarih')['Miktar'].sum().reset_index()

print(df_filtered.info())
# Yıllık toplam satışları görselleştirme
plt.figure(figsize=(15, 5))
plt.plot(yearly_sales['Tarih'], yearly_sales['Miktar'], marker='o')
plt.title('Yıllık Toplam Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış (Miktar)')
plt.grid(True)
plt.show()

print(df_filtered.info())
print(df_filtered.head(10))

#%% EN ÇOK SATILAN ÜRÜN YILLIK
en_cok_satis = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün satış miktarını aylık olarak grupla
df_en_cok_satis = df_filtered[df_filtered["Ürün Kodu"] == en_cok_satis].resample('Y', on='Tarih')['Miktar'].sum().reset_index()

print(df_en_cok_satis.info())
print(df_en_cok_satis.head(20))

# Grafik boyutunu ve yazı tipini ayarla
plt.figure(figsize=(15, 5), dpi=120)
plt.rcParams["font.size"] = 14

# Çizgi grafiği çiz
plt.plot(df_en_cok_satis['Tarih'], df_en_cok_satis['Miktar'], marker='o', color="green", linestyle='-', linewidth=2)
plt.title(f"{en_cok_satis} Ürününün Yıllık Satışları")
plt.xlabel('Tarih')
plt.ylabel('Satış Miktarı')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#%%
en_cok_satis_m = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]

# En çok satılan ürünün tüm sütunlarını aylık olarak grupla
df_en_cok_satis = df_filtered[df_filtered["Ürün Kodu"] == en_cok_satis_m].resample('M', on='Tarih').sum().reset_index()

df_train = df_en_cok_satis.copy()

print(df_train)
print(df_train.info())

df_train= df_train.dropna()

print(df_train.head(15))

df_train = df_train.drop(['Yıl', 'Toplam', 'Net Tutar', 'Ürün Kodu'], axis=1)

print(df_train.info())


#%%
from prophet import Prophet

print(df_train.info())

df_train['Müşteri Kodu'] = df_train['Müşteri Kodu'].astype('category').cat.codes

df_train.rename(columns={'Tarih': 'ds', 'Miktar': 'y'}, inplace=True)


#%% PROPHET

model = Prophet()

model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 'Müşteri Kodu' ve 'Birim Fiyat' sütunlarını regresör olarak ekleyin
model.add_regressor('Müşteri Kodu')
model.add_regressor('Birim Fiyat')


model.fit(df_train)

future = model.make_future_dataframe(periods=365)
future.tail()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = model.plot(forecast)

fig2 = model.plot_components(forecast)

#%%

print(df_train.head())

print(df_filtered.head())

#%% en çok satan ürün için arima

from statsmodels.tsa.arima.model import ARIMA
pd.set_option("display.max_columns", None)

df_arima = df_train.resample('M').sum()

print(df_arima.head(20))
# print(df_filtered.head(50))

# Veri setini eğitim ve test olarak ayır
train_size = int(len(df_arima) * 0.80)
train, test = df_arima[:train_size], df_arima[train_size:]

# ARIMA modelini uygula
order = (5, 1, 0)  # Örnek bir order (p, d, q) değeri
model = ARIMA(train, order=order)
model_fit = model.fit()

# Tahmin yap
forecast_steps = len(test)
forecast = model_fit.get_forecast(steps=forecast_steps)
predicted_values = forecast.predicted_mean

# Tahmin ve gerçek değerleri karşılaştır
plt.plot(train.index, train, label='Eğitim Verisi')
plt.plot(test.index, test, label='Test Verisi')
plt.plot(test.index, predicted_values, label='ARIMA Tahmini')
plt.legend()
plt.show()

#%% ARIMA toplam tüm ürünler için

df_arima_2 = df_filtered.copy()

print(df_arima_2.head())

df_arima_2 = df_arima_2.drop(['Yıl', 'Müşteri Kodu', 'Ürün Kodu','Toplam','Birim Fiyat' , 'Net Tutar'] , axis=1)

df_arima_2['Tarih'] = pd.to_datetime(df_arima_2['Tarih'])
df_arima_2.set_index('Tarih', inplace=True)
df_arima_2 = df_arima_2.resample('M').sum()


print(df_arima_2.head(20))

train_size = int(len(df_arima_2) * 0.80)
train, test = df_arima_2[:train_size], df_arima_2[train_size:]

# ARIMA modelini uygula
order = (5, 1, 0)  # Örnek bir order (p, d, q) değeri
model = ARIMA(train, order=order)
model_fit = model.fit()

# Tahmin yap
forecast_steps = len(test)
forecast = model_fit.get_forecast(steps=forecast_steps)
predicted_values = forecast.predicted_mean

# Tahmin ve gerçek değerleri karşılaştır
plt.plot(train.index, train, label='Eğitim Verisi')
plt.plot(test.index, test, label='Test Verisi')
plt.plot(test.index, predicted_values, label='ARIMA Tahmini')
plt.legend()
plt.show()

#%% 'Net Tutar' ve 'Birim Fiyat' sütunları eklendi (NET TUTAR İÇİN TAHMİN) -En çok satan ürün için

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

df_sarima = df_en_cok_satis.copy()
print(df_sarima.info())

df_sarima = df_sarima.drop(['Yıl' ,'Toplam', 'Ürün Kodu', 'Müşteri Kodu'] , axis=1)
print(df_sarima.info())

df_sarima.set_index('Tarih', inplace=True)
df_sarima = df_sarima.resample('M').sum()  # Günlük toplamlar için örnekleme

# 'Net Tutar' sütunu ile dummy değişkenlerini içeren yeni bir veri çerçevesi oluştur
df_sarima = df_sarima[['Net Tutar', 'Birim Fiyat']]

# Veri setini eğitim ve test olarak ayır
train_size = int(len(df_sarima) * 0.8)
train, test = df_sarima[:train_size], df_sarima[train_size:]

# SARIMA modelini uygula
order = (1, 1, 1)  # Örnek bir order değeri
exog_train, exog_test = train[['Birim Fiyat']], test[['Birim Fiyat']]
model_net = SARIMAX(endog=train['Net Tutar'], exog=exog_train, order=(1, 1, 1))
results = model_net.fit()

# Tahmin yap
forecast_steps = len(test)
forecast = results.get_forecast(steps=forecast_steps, exog=exog_test)
predicted_values = forecast.predicted_mean

# Tahmin ve gerçek değerleri karşılaştır
plt.plot(train.index, train['Net Tutar'], label='Eğitim Verisi')
plt.plot(test.index, test['Net Tutar'], label='Test Verisi')
plt.plot(test.index, predicted_values, label='SARIMA Tahmini')
plt.legend()
plt.show()

print(df_sarima.head(5))

#%%  SARIMA

df_sarima = df_en_cok_satis.copy()
print(df_sarima.info())

# 'Müşteri Kodu' sütununu kaldır
# df_sarima = df_sarima.drop(['Yıl', 'Toplam', 'Ürün Kodu', 'Müşteri Kodu'], axis=1)
print(df_sarima.info())

df_sarima.set_index('Tarih', inplace=True)
df_sarima = df_sarima.resample('M').sum()  # Günlük toplamlar için örnekleme

# 'Net Tutar', 'Birim Fiyat' ve 'Miktar' sütunlarını içeren yeni bir veri çerçevesi oluştur
df_sarima = df_sarima[['Miktar', 'Birim Fiyat']]

# Veriyi eğitim ve test olarak ayırın
train_size = int(len(df_sarima) * 0.8)
train, test = df_sarima[:train_size], df_sarima[train_size:]

# SARIMA modelini uygula
order = (1, 1, 1)  # Örnek bir order değeri
exog_train, exog_test = train[['Birim Fiyat']], test[['Birim Fiyat']]
model_miktar = SARIMAX(endog=train['Miktar'], exog=exog_train, order=order)
results = model_miktar.fit()

# Tahmin yap
forecast_steps = len(test)
forecast = results.get_forecast(steps=forecast_steps, exog=exog_test)
predicted_values = forecast.predicted_mean

# Tahmin ve gerçek değerleri karşılaştır
plt.plot(train.index, train['Miktar'], label='Eğitim Verisi')
plt.plot(test.index, test['Miktar'], label='Test Verisi')
plt.plot(test.index, predicted_values, label='SARIMA Tahmini (Miktar)')
plt.legend()
plt.show()

#%% Random forest

print(df_filtered.head())

df_week = df_filtered.copy()

df_week['Tarih'] = pd.to_datetime(df_week['Tarih'])

# Tarihlere göre sıralama yapın
df_week = df_week.sort_values('Tarih')

# Hafta numaralarını hesaplayın
df_week['Hafta'] = (df_week['Tarih'] - df_week['Tarih'].min()).dt.days // 7 + 1

df_week['Ay'] = (df_week['Tarih'].dt.year - df_week['Tarih'].dt.year.min()) * 12 + df_week['Tarih'].dt.month

print(df_week.head(-5))


melt = df_week.copy()
# 'Product_Code' ve 'Week' sütunlarındaki sayısal değerleri çıkartın
melt['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt['Month'] = df_week['Ay']
melt['Week'] = df_week['Hafta']
melt['Sales'] = melt['Miktar']

melt = melt.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis = 1)

# 'Week' ve 'Product_Code' sütunlarına göre sıralayın
melt = melt.sort_values(['Month', 'Product_Code'])

product_num = melt['Product_Code'].nunique()
print("Farklı Ürün Sayısı", product_num)
# Sonucu görüntüleyin
print(melt.info())
label_encoder = LabelEncoder()
# 'Product_Code' sütununu dönüştürün
melt['Product_Code'] = label_encoder.fit_transform(melt['Product_Code']) + 1  # +1 ekleyerek 0'dan değil, 1'den başlatın

print(melt.head())

split_point = 65

melt_train = melt[melt['Month'] < split_point].copy()
melt_valid = melt[melt['Month'] >= split_point].copy()

print(melt_train.head())
print(melt_valid.head())

#%%

melt_train['sales_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid['sales_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)

print(melt_train.head())


melt_train = melt_train.dropna()

print(melt_train.tail())

melt_train["lag_sales_1"] = melt_train.groupby("Product_Code")['Sales'].shift(1)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["lag_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].shift(1)



#%% Difference

melt_train["diff_sales_1"] = melt_train.groupby("Product_Code")['Sales'].diff(1)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["diff_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].diff(1)


#%% Rolling statistics

melt_train.groupby("Product_Code")['Sales'].rolling(4).mean()

melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

melt_train["mean_sales_4"] = melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["mean_sales_4"] = melt_valid.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)


#%%

def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    #ape[~np.isfinite(ape)] = 0. # VERY questionable
    ape[~np.isfinite(ape)] = 1. # pessimist estimate
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


#%%Establish baseline



y_pred = melt_train['Sales']
y_true = melt_train['sales_next_month']

print(mape(y_true, y_pred))
print(wmape(y_true, y_pred))


#%% TRAIN
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

print(melt_train.head(5))

features = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']

imputer = SimpleImputer()
Xtr = imputer.fit_transform(melt_train[features])
ytr = melt_train['sales_next_month']

mdl = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
mdl.fit(Xtr, ytr)

Xval = imputer.transform(melt_valid[features])
yval = melt_valid['sales_next_month']

p = mdl.predict(Xval)

print(mape(yval, p))
print(wmape(yval, p))

melt_train['sales_next_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-2)
melt_valid['sales_next_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-2)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_train = melt_train.dropna(subset=['sales_next_month','sales_next_next_month'])

imputer = SimpleImputer()
Xtr = imputer.fit_transform(melt_train[features])
ytr = melt_train[['sales_next_month', 'sales_next_next_month']]

mdl = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
mdl.fit(Xtr, ytr)

Xval = imputer.transform(melt_valid[features])
yval = melt_valid[['sales_next_month', 'sales_next_next_month']]

p = mdl.predict(Xval)


print(mape(yval, p))
print(wmape(yval, p))


print(melt_valid.tail())
new_examples = melt_valid[melt_valid['Month'] == 80].copy()

# Check for NaN values in new_examples
nan_values_before_imputation = new_examples.isna().sum()
print("NaN values before imputation:\n", nan_values_before_imputation)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy
new_examples_imputed = pd.DataFrame(imputer.fit_transform(new_examples), columns=new_examples.columns)


# Ensure features are consistent
features = [feature for feature in features if feature in new_examples_imputed.columns]

# Make predictions
p = mdl.predict(new_examples_imputed[features])

# Create a DataFrame to store predictions
predictions_df = new_examples_imputed.copy()

# Add predictions as new columns
predictions_df['p_sales_next_month'] = p[:, 0]
predictions_df['p_sales_next_next_month'] = p[:, 1]
#%%
print(predictions_df.head(10))

#%% Linear Regression
#%% with Lineer Regresyon
from sklearn.impute import SimpleImputer

melt2 = df_week.copy()
melt2['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt2['Month'] = df_week['Ay']
melt2['Week'] = df_week['Hafta']
melt2['Sales'] = melt2['Miktar']
melt2 = melt2.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

label_encoder = LabelEncoder()
melt2['Product_Code'] = label_encoder.fit_transform(melt2['Product_Code']) + 1

split_point = 65
melt_train = melt2[melt2['Month'] < split_point].copy()
melt_valid = melt2[melt2['Month'] >= split_point].copy()

melt_train['sales_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
melt_valid['sales_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)
melt_train = melt_train.dropna()


melt_train["lag_sales_1"] = melt_train.groupby("Product_Code")['Sales'].shift(1)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["lag_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].shift(1)


melt_train["diff_sales_1"] = melt_train.groupby("Product_Code")['Sales'].diff(1)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["diff_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].diff(1)

# Rolling statistics

melt_train.groupby("Product_Code")['Sales'].rolling(4).mean()

melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

melt_train["mean_sales_4"] = melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

print(melt_train[melt_train['Product_Code'] == 1].head())

melt_valid["mean_sales_4"] = melt_valid.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)



# Özelliklerin ve hedef değişkenin seçilmesi
features = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
target = 'sales_next_month'

# Eğitim seti için özellik ve hedef matrisleri
X_train = melt_train[features]
y_train = melt_train[target]
X_valid = melt_valid[features]
y_valid = melt_valid[target]


imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)
X_valid_imputed = imputer.transform(X_valid)

# Lineer Regresyon modeli
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_imputed, y_train)

# RandomForestRegressor modeli
rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
rf_model.fit(X_train_imputed, y_train)

# Değerlendirme
predictions_linear_reg = linear_reg_model.predict(X_valid_imputed)
predictions_rf = rf_model.predict(X_valid_imputed)
# Değerlendirme


# Performans metrikleri
mape_linear_reg = mape(y_valid, predictions_linear_reg)
wmape_linear_reg = wmape(y_valid, predictions_linear_reg)

mape_rf = mape(y_valid, predictions_rf)
wmape_rf = wmape(y_valid, predictions_rf)

print("Lineer Regresyon MAPE:", mape_linear_reg)
print("Lineer Regresyon WMAPE:", wmape_linear_reg)

print("RandomForestRegressor MAPE:", mape_rf)
print("RandomForestRegressor WMAPE:", wmape_rf)

# Gelecekteki tahminler

new_examples = melt_valid[melt_valid['Month'] == 80].copy()
new_examples_imputed = pd.DataFrame(imputer.fit_transform(new_examples[features]), columns=new_examples.columns)

new_examples = melt_valid[melt_valid['Month'] == 80].copy()

# Check for NaN values in new_examples
nan_values_before_imputation = new_examples.isna().sum()
print("NaN values before imputation:\n", nan_values_before_imputation)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy
new_examples_imputed = pd.DataFrame(imputer.fit_transform(new_examples), columns=new_examples.columns)

# Ensure features are consistent
features = [feature for feature in features if feature in new_examples_imputed.columns]

# Lineer Regresyon için gelecekteki tahminler
predictions_linear_reg_future = linear_reg_model.predict(new_examples_imputed[features])

# RandomForestRegressor için gelecekteki tahminler
predictions_rf_future = rf_model.predict(new_examples_imputed[features])

# Tahminleri DataFrame'e ekleme
new_examples['p_sales_next_month_linear_reg'] = predictions_linear_reg_future
new_examples['p_sales_next_month_rf'] = predictions_rf_future

print(new_examples[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_linear_reg', 'p_sales_next_month_rf']])


#%% XGBoost
#%%
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# XGBoost için gerekli kütüphaneyi ekleyin
from xgboost import XGBRegressor

# Veri setini kopyalayın
melt_xgb = df_week.copy()

# 'Product_Code' ve 'Week' sütunlarındaki sayısal değerleri çıkartın
melt_xgb['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt_xgb['Month'] = df_week['Ay']
melt_xgb['Week'] = df_week['Hafta']
melt_xgb['Sales'] = melt_xgb['Miktar']

melt_xgb = melt_xgb.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

# 'Product_Code' sütununu dönüştürün
melt_xgb['Product_Code'] = label_encoder.fit_transform(melt_xgb['Product_Code']) + 1

# Eğitim ve geçerleme setlerini ayırın
split_point_xgb = 65
melt_train_xgb = melt_xgb[melt_xgb['Month'] < split_point_xgb].copy()
melt_valid_xgb = melt_xgb[melt_xgb['Month'] >= split_point_xgb].copy()

# Hedef değişkeni belirleyin
target_xgb = 'sales_next_month'

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_xgb[target_xgb] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_xgb[target_xgb] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_xgb = melt_train_xgb.dropna()


melt_train_xgb['sales_next_month'] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(-1)
print(melt_train_xgb[melt_train_xgb['Product_Code'] == 1].head())

melt_valid_xgb['sales_next_month'] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(-1)

print(melt_train_xgb.head())


melt_train_xgb = melt_train_xgb.dropna()

print(melt_train_xgb.tail())

melt_train_xgb["lag_sales_1"] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(1)

print(melt_train_xgb[melt_train_xgb['Product_Code'] == 1].head())

melt_valid_xgb["lag_sales_1"] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(1)



melt_train_xgb["diff_sales_1"] = melt_train_xgb.groupby("Product_Code")['Sales'].diff(1)

print(melt_train_xgb[melt_train_xgb['Product_Code'] == 1].head())

melt_valid_xgb["diff_sales_1"] = melt_valid_xgb.groupby("Product_Code")['Sales'].diff(1)



melt_train_xgb.groupby("Product_Code")['Sales'].rolling(4).mean()

melt_train_xgb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

melt_train_xgb["mean_sales_4"] = melt_train_xgb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

print(melt_train_xgb[melt_train_xgb['Product_Code'] == 1].head())

melt_valid_xgb["mean_sales_4"] = melt_valid_xgb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)


def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    #ape[~np.isfinite(ape)] = 0. # VERY questionable
    ape[~np.isfinite(ape)] = 1. # pessimist estimate
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# Özelliklerin ve hedef değişkenin seçilmesi
features_xgb = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']

# Eğitim seti için özellik ve hedef matrisleri
X_train_xgb = melt_train_xgb[features_xgb]
y_train_xgb = melt_train_xgb[target_xgb]
X_valid_xgb = melt_valid_xgb[features_xgb]
y_valid_xgb = melt_valid_xgb[target_xgb]

# Eksik değerleri doldurun
X_train_imputed_xgb = imputer.fit_transform(X_train_xgb)
X_valid_imputed_xgb = imputer.transform(X_valid_xgb)

# XGBoost Regresyon modeli
xgb_model = XGBRegressor(n_estimators=100, random_state=0, n_jobs=6)
xgb_model.fit(X_train_imputed_xgb, y_train_xgb)

# Geçerleme seti üzerinde tahminler yapın
y_pred_xgb = xgb_model.predict(X_valid_imputed_xgb)

# Performans metrikleri
mape_xgb = mape(y_valid_xgb, y_pred_xgb)
wmape_xgb = wmape(y_valid_xgb, y_pred_xgb)

print("XGBoost MAPE:", mape_xgb)
print("XGBoost WMAPE:", wmape_xgb)

# Gelecekteki tahminler
new_examples_xgb = melt_valid_xgb[melt_valid_xgb['Month'] == 81].copy()
new_examples_imputed_xgb = pd.DataFrame(imputer.fit_transform(new_examples_xgb[features_xgb]), columns=features_xgb)

# XGBoost için gelecekteki tahminler
predictions_xgb_future = xgb_model.predict(new_examples_imputed_xgb)

# Tahminleri DataFrame'e ekleme
new_examples_xgb['p_sales_next_month_xgb'] = predictions_xgb_future

print(new_examples_xgb[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_xgb']])


#%% LightGBM
#%%
import lightgbm as lgb

# Veri setini kopyalayın
melt_gbm = df_week.copy()

# 'Product_Code' ve 'Week' sütunlarındaki sayısal değerleri çıkartın
melt_gbm['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt_gbm['Month'] = df_week['Ay']
melt_gbm['Week'] = df_week['Hafta']
melt_gbm['Sales'] = melt_gbm['Miktar']

melt_gbm = melt_gbm.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

# 'Product_Code' sütununu dönüştürün
melt_gbm['Product_Code'] = label_encoder.fit_transform(melt_gbm['Product_Code']) + 1

# Eğitim ve geçerleme setlerini ayırın
split_point_gbm = 65
melt_train_gbm = melt_gbm[melt_gbm['Month'] < split_point_gbm].copy()
melt_valid_gbm = melt_gbm[melt_gbm['Month'] >= split_point_gbm].copy()

# Hedef değişkeni belirleyin
target_gbm = 'sales_next_month'

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_gbm[target_gbm] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_gbm[target_gbm] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_gbm = melt_train_gbm.dropna()

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_gbm[target_gbm] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_gbm[target_gbm] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_gbm = melt_train_gbm.dropna()

melt_train_gbm['sales_next_month'] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(-1)
print(melt_train_gbm[melt_train_gbm['Product_Code'] == 1].head())

melt_valid_gbm['sales_next_month'] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(-1)

print(melt_train_gbm.head())

melt_train_gbm = melt_train_gbm.dropna()

print(melt_train_gbm.tail())

melt_train_gbm["lag_sales_1"] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(1)

print(melt_train_gbm[melt_train_gbm['Product_Code'] == 1].head())

melt_valid_gbm["lag_sales_1"] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(1)

melt_train_gbm["diff_sales_1"] = melt_train_gbm.groupby("Product_Code")['Sales'].diff(1)

print(melt_train_gbm[melt_train_gbm['Product_Code'] == 1].head())

melt_valid_gbm["diff_sales_1"] = melt_valid_gbm.groupby("Product_Code")['Sales'].diff(1)

melt_train_gbm.groupby("Product_Code")['Sales'].rolling(4).mean()

melt_train_gbm.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

melt_train_gbm["mean_sales_4"] = melt_train_gbm.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

print(melt_train_gbm[melt_train_gbm['Product_Code'] == 1].head())

melt_valid_gbm["mean_sales_4"] = melt_valid_gbm.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

# Özelliklerin ve hedef değişkenin seçilmesi
features_gbm = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
target_gbm = 'sales_next_month'

# Eğitim seti için özellik ve hedef matrisleri
X_train_gbm = melt_train_gbm[features_gbm]
y_train_gbm = melt_train_gbm[target_gbm]
X_valid_gbm = melt_valid_gbm[features_gbm]
y_valid_gbm = melt_valid_gbm[target_gbm]

# Eksik değerleri doldurun
X_train_imputed_gbm = imputer.fit_transform(X_train_gbm)
X_valid_imputed_gbm = imputer.transform(X_valid_gbm)

# LightGBM Regresyon modeli
gbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=0, n_jobs=6)
gbm_model.fit(X_train_imputed_gbm, y_train_gbm)

# Geçerleme seti üzerinde tahminler yapın
y_pred_gbm = gbm_model.predict(X_valid_imputed_gbm)

# Performans metrikleri
mape_gbm = mape(y_valid_gbm, y_pred_gbm)
wmape_gbm = wmape(y_valid_gbm, y_pred_gbm)

print("LightGBM MAPE:", mape_gbm)
print("LightGBM WMAPE:", wmape_gbm)

# Gelecekteki tahminler
new_examples_gbm = melt_valid_gbm[melt_valid_gbm['Month'] == 81].copy()
new_examples_imputed_gbm = pd.DataFrame(imputer.fit_transform(new_examples_gbm[features_gbm]), columns=features_gbm)

# LightGBM için gelecekteki tahminler
predictions_gbm_future = gbm_model.predict(new_examples_imputed_gbm)

# Tahminleri DataFrame'e ekleme
new_examples_gbm['p_sales_next_month_gbm'] = predictions_gbm_future

print(new_examples_gbm[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_gbm']])

"""
Aylık olarak ürün bazındaki satışlar grupla. tahmin edilen toplam satışları elde et. görselleştirme yap.
Hiperparametreleri ekle
hepsini aynı tabloda topla

"""
#%% gradient
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Veri setini kopyalayın
melt_gb = df_week.copy()

# 'Product_Code' ve 'Week' sütunlarındaki sayısal değerleri çıkartın
melt_gb['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt_gb['Month'] = df_week['Ay']
melt_gb['Week'] = df_week['Hafta']
melt_gb['Sales'] = melt_gb['Miktar']

melt_gb = melt_gb.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

# 'Product_Code' sütununu dönüştürün
melt_gb['Product_Code'] = label_encoder.fit_transform(melt_gb['Product_Code']) + 1

# Eğitim ve geçerleme setlerini ayırın
split_point_gb = 65
melt_train_gb = melt_gb[melt_gb['Month'] < split_point_gb].copy()
melt_valid_gb = melt_gb[melt_gb['Month'] >= split_point_gb].copy()

# Hedef değişkeni belirleyin
target_gb = 'sales_next_month'

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_gb[target_gb] = melt_train_gb.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_gb[target_gb] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_gb = melt_train_gb.dropna()

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_gb[target_gb] = melt_train_gb.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_gb[target_gb] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_gb = melt_train_gb.dropna()

melt_train_gb['sales_next_month'] = melt_train_gb.groupby("Product_Code")['Sales'].shift(-1)
print(melt_train_gb[melt_train_gb['Product_Code'] == 1].head())

melt_valid_gb['sales_next_month'] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(-1)

print(melt_train_gb.head())

melt_train_gb = melt_train_gb.dropna()

print(melt_train_gb.tail())

melt_train_gb["lag_sales_1"] = melt_train_gb.groupby("Product_Code")['Sales'].shift(1)

print(melt_train_gb[melt_train_gb['Product_Code'] == 1].head())

melt_valid_gb["lag_sales_1"] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(1)

melt_train_gb["diff_sales_1"] = melt_train_gb.groupby("Product_Code")['Sales'].diff(1)

print(melt_train_gb[melt_train_gb['Product_Code'] == 1].head())

melt_valid_gb["diff_sales_1"] = melt_valid_gb.groupby("Product_Code")['Sales'].diff(1)

melt_train_gb.groupby("Product_Code")['Sales'].rolling(4).mean()

melt_train_gb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

melt_train_gb["mean_sales_4"] = melt_train_gb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

print(melt_train_gb[melt_train_gb['Product_Code'] == 1].head())

melt_valid_gb["mean_sales_4"] = melt_valid_gb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

# Özelliklerin ve hedef değişkenin seçilmesi
features_gb = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
target_gb = 'sales_next_month'

# Eğitim seti için özellik ve hedef matrisleri
X_train_gb = melt_train_gb[features_gb]
y_train_gb = melt_train_gb[target_gb]
X_valid_gb = melt_valid_gb[features_gb]
y_valid_gb = melt_valid_gb[target_gb]

# Eksik değerleri doldurun
X_train_imputed_gb = imputer.fit_transform(X_train_gb)
X_valid_imputed_gb = imputer.transform(X_valid_gb)

# Gradient Boosting Regresyon modeli
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
gb_model.fit(X_train_imputed_gb, y_train_gb)

# Geçerleme seti üzerinde tahminler yapın
y_pred_gb = gb_model.predict(X_valid_imputed_gb)

# Performans metrikleri
mape_gb = mape(y_valid_gb, y_pred_gb)
wmape_gb = wmape(y_valid_gb, y_pred_gb)

print("Gradient Boosting MAPE:", mape_gb)
print("Gradient Boosting WMAPE:", wmape_gb)

# Gelecekteki tahminler
new_examples_gb = melt_valid_gb[melt_valid_gb['Month'] == 81].copy()
new_examples_imputed_gb = pd.DataFrame(imputer.fit_transform(new_examples_gb[features_gb]), columns=features_gb)

# Gradient Boosting için gelecekteki tahminler
predictions_gb_future = gb_model.predict(new_examples_imputed_gb)

# Tahminleri DataFrame'e ekleme
new_examples_gb['p_sales_next_month_gb'] = predictions_gb_future

print(new_examples_gb[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_gb']])

#%% LSTM

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Veri setini kopyalayın
melt_lstm = df_week.copy()

# 'Product_Code' ve 'Week' sütunlarındaki sayısal değerleri çıkartın
melt_lstm['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt_lstm['Month'] = df_week['Ay']
melt_lstm['Week'] = df_week['Hafta']
melt_lstm['Sales'] = melt_lstm['Miktar']

melt_lstm = melt_lstm.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

# 'Product_Code' sütununu dönüştürün
melt_lstm['Product_Code'] = label_encoder.fit_transform(melt_lstm['Product_Code']) + 1

# Eğitim ve geçerleme setlerini ayırın
split_point_lstm = 65
melt_train_lstm = melt_lstm[melt_lstm['Month'] < split_point_lstm].copy()
melt_valid_lstm = melt_lstm[melt_lstm['Month'] >= split_point_lstm].copy()

# Hedef değişkeni belirleyin
target_lstm = 'sales_next_month'

# Hedef değişkenin gelecekteki değerini ekleyin
melt_train_lstm[target_lstm] = melt_train_lstm.groupby("Product_Code")['Sales'].shift(-1)
melt_valid_lstm[target_lstm] = melt_valid_lstm.groupby("Product_Code")['Sales'].shift(-1)

# NaN değerleri düşürün
melt_train_lstm = melt_train_lstm.dropna()

# LSTM için uygun formatı oluşturun
def create_lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Özellik ve hedef değişkenini seçin
features_lstm = ['Sales', 'Product_Code', 'Month', 'Week']
target_lstm = 'sales_next_month'

# LSTM için uygun formatı oluşturun
look_back = 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(melt_train_lstm[features_lstm + [target_lstm]].values)
dataset_valid = scaler.transform(melt_valid_lstm[features_lstm + [target_lstm]].values)

trainX, trainY = create_lstm_dataset(dataset_train, look_back)
validX, validY = create_lstm_dataset(dataset_valid, look_back)

# LSTM modeli oluşturun
model_lstm = Sequential()
model_lstm.add(LSTM(50, input_shape=(look_back, len(features_lstm) + 1)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')

# Erken durdurma callback'i
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# LSTM modelini eğitin
model_lstm.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(validX, validY), callbacks=[early_stopping], verbose=2)

# Geçerleme seti üzerinde tahminler yapın
trainPredict = model_lstm.predict(trainX)
validPredict = model_lstm.predict(validX)

# Tahminleri orijinal ölçeklendirmeye geri dönüştürün
trainPredict = scaler.inverse_transform(np.concatenate((trainPredict, np.zeros((len(trainPredict), len(features_lstm))), np.zeros((len(trainPredict), 1))), axis=1))[:, 0]
trainY = scaler.inverse_transform(np.concatenate((trainY.reshape(-1, 1), np.zeros((len(trainY), len(features_lstm))), np.zeros((len(trainY), 1))), axis=1))[:, 0]
validPredict = scaler.inverse_transform(np.concatenate((validPredict, np.zeros((len(validPredict), len(features_lstm))), np.zeros((len(validPredict), 1))), axis=1))[:, 0]
validY = scaler.inverse_transform(np.concatenate((validY.reshape(-1, 1), np.zeros((len(validY), len(features_lstm))), np.zeros((len(validY), 1))), axis=1))[:, 0]

# Performans metrikleri
mae_train_lstm = mean_absolute_error(trainY, trainPredict)
mae_valid_lstm = mean_absolute_error(validY, validPredict)

print("LSTM MAE (Training):", mae_train_lstm)
print("LSTM MAE (Validation):", mae_valid_lstm)

# Gelecekteki tahminler
new_examples_lstm = melt_valid_lstm[melt_valid_lstm['Month'] == 81].copy()
new_examples_lstm_scaled = scaler.transform(new_examples_lstm[features_lstm + [target_lstm]].values)
new_examples_lstmX, _ = create_lstm_dataset(new_examples_lstm_scaled, look_back)

# LSTM için gelecekteki tahminler
predictions_lstm_future_scaled = model_lstm.predict(new_examples_lstmX)
predictions_lstm_future = scaler.inverse_transform(np.concatenate((predictions_lstm_future_scaled, np.zeros((len(predictions_lstm_future_scaled), len(features_lstm))), np.zeros((len(predictions_lstm_future_scaled), 1))), axis=1))[:, 0]

# Tahminleri DataFrame'e ekleme
new_examples_lstm['p_sales_next_month_lstm'] = predictions_lstm_future

print(new_examples_lstm[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_lstm']])

#%% PROPHET

import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot

print(df_filtered.head())

import fbprophet
print('Prophet %s' % fbprophet.__version__)

pro_selected = df_filtered.groupby("Ürün Kodu")["Miktar"].sum().sort_values(ascending=False).index[0]
df_pro_selected = df_filtered[df_filtered["Ürün Kodu"] == pro_selected]


#%%%

def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    #ape[~np.isfinite(ape)] = 0. # VERY questionable
    ape[~np.isfinite(ape)] = 1. # pessimist estimate
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def train_and_evaluate_model(df_week):
    melt2 = df_week.copy()
    melt2['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
    melt2['Month'] = df_week['Ay']
    melt2['Week'] = df_week['Hafta']
    melt2['Sales'] = melt2['Miktar']
    melt2 = melt2.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

    label_encoder = LabelEncoder()
    melt2['Product_Code'] = label_encoder.fit_transform(melt2['Product_Code']) + 1

    split_point = 65
    melt_train = melt2[melt2['Month'] < split_point].copy()
    melt_valid = melt2[melt2['Month'] >= split_point].copy()

    melt_train['sales_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid['sales_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)
    melt_train = melt_train.dropna()

    melt_train["lag_sales_1"] = melt_train.groupby("Product_Code")['Sales'].shift(1)

    melt_valid["lag_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].shift(1)

    melt_train["diff_sales_1"] = melt_train.groupby("Product_Code")['Sales'].diff(1)

    melt_valid["diff_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].diff(1)

    melt_train["mean_sales_4"] = melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    melt_valid["mean_sales_4"] = melt_valid.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    features = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
    target = 'sales_next_month'

    X_train = melt_train[features]
    y_train = melt_train[target]
    X_valid = melt_valid[features]
    y_valid = melt_valid[target]

    imputer = SimpleImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train_imputed, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
    rf_model.fit(X_train_imputed, y_train)

    predictions_linear_reg = linear_reg_model.predict(X_valid_imputed)
    predictions_rf = rf_model.predict(X_valid_imputed)

    mape_linear_reg = mape(y_valid, predictions_linear_reg)
    wmape_linear_reg = wmape(y_valid, predictions_linear_reg)

    mape_rf = mape(y_valid, predictions_rf)
    wmape_rf = wmape(y_valid, predictions_rf)

    print("Lineer Regresyon MAPE:", mape_linear_reg)
    print("Lineer Regresyon WMAPE:", wmape_linear_reg)

    print("RandomForestRegressor MAPE:", mape_rf)
    print("RandomForestRegressor WMAPE:", wmape_rf)

    new_examples = melt_valid[melt_valid['Month'] == 80].copy()
    new_examples_imputed = pd.DataFrame(imputer.fit_transform(new_examples[features]), columns=features)

    features = [feature for feature in features if feature in new_examples_imputed.columns]

    predictions_linear_reg_future = linear_reg_model.predict(new_examples_imputed[features])
    predictions_rf_future = rf_model.predict(new_examples_imputed[features])

    new_examples['p_sales_next_month_linear_reg'] = predictions_linear_reg_future
    new_examples['p_sales_next_month_rf'] = predictions_rf_future

    print(new_examples[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_linear_reg', 'p_sales_next_month_rf']])
    

print(df_week.head())
train_and_evaluate_model(df_week)

#%%
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate_gb_model(df_week):

    melt_gb = df_week.copy()

    melt_gb['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
    melt_gb['Month'] = df_week['Ay']
    melt_gb['Week'] = df_week['Hafta']
    melt_gb['Sales'] = melt_gb['Miktar']

    melt_gb = melt_gb.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

    melt_gb['Product_Code'] = label_encoder.fit_transform(melt_gb['Product_Code']) + 1

    split_point_gb = 65
    melt_train_gb = melt_gb[melt_gb['Month'] < split_point_gb].copy()
    melt_valid_gb = melt_gb[melt_gb['Month'] >= split_point_gb].copy()

    target_gb = 'sales_next_month'

    melt_train_gb[target_gb] = melt_train_gb.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid_gb[target_gb] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_gb = melt_train_gb.dropna()

    melt_train_gb['sales_next_month'] = melt_train_gb.groupby("Product_Code")['Sales'].shift(-1)

    melt_valid_gb['sales_next_month'] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_gb = melt_train_gb.dropna()

    melt_train_gb["lag_sales_1"] = melt_train_gb.groupby("Product_Code")['Sales'].shift(1)

    melt_valid_gb["lag_sales_1"] = melt_valid_gb.groupby("Product_Code")['Sales'].shift(1)

    melt_train_gb["diff_sales_1"] = melt_train_gb.groupby("Product_Code")['Sales'].diff(1)

    melt_valid_gb["diff_sales_1"] = melt_valid_gb.groupby("Product_Code")['Sales'].diff(1)

    melt_train_gb["mean_sales_4"] = melt_train_gb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    melt_valid_gb["mean_sales_4"] = melt_valid_gb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    features_gb = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
    target_gb = 'sales_next_month'

    X_train_gb = melt_train_gb[features_gb]
    y_train_gb = melt_train_gb[target_gb]
    X_valid_gb = melt_valid_gb[features_gb]
    y_valid_gb = melt_valid_gb[target_gb]

    imputer = SimpleImputer()
    X_train_imputed_gb = imputer.fit_transform(X_train_gb)
    X_valid_imputed_gb = imputer.transform(X_valid_gb)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
    gb_model.fit(X_train_imputed_gb, y_train_gb)

    y_pred_gb = gb_model.predict(X_valid_imputed_gb)

    mape_gb = mape(y_valid_gb, y_pred_gb)
    wmape_gb = wmape(y_valid_gb, y_pred_gb)

    print("Gradient Boosting MAPE:", mape_gb)
    print("Gradient Boosting WMAPE:", wmape_gb)

    new_examples_gb = melt_valid_gb[melt_valid_gb['Month'] == 81].copy()
    new_examples_imputed_gb = pd.DataFrame(imputer.fit_transform(new_examples_gb[features_gb]), columns=features_gb)

    predictions_gb_future = gb_model.predict(new_examples_imputed_gb)

    new_examples_gb['p_sales_next_month_gb'] = predictions_gb_future

    print(new_examples_gb[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_gb']])

# Örnek Kullanım:
train_and_evaluate_gb_model(df_week)

#%%

import lightgbm as lgb

def train_and_evaluate_gbm_model(df_week):
    melt_gbm = df_week.copy()

    melt_gbm['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
    melt_gbm['Month'] = df_week['Ay']
    melt_gbm['Week'] = df_week['Hafta']
    melt_gbm['Sales'] = melt_gbm['Miktar']

    melt_gbm = melt_gbm.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

    melt_gbm['Product_Code'] = label_encoder.fit_transform(melt_gbm['Product_Code']) + 1

    split_point_gbm = 65
    melt_train_gbm = melt_gbm[melt_gbm['Month'] < split_point_gbm].copy()
    melt_valid_gbm = melt_gbm[melt_gbm['Month'] >= split_point_gbm].copy()

    target_gbm = 'sales_next_month'

    melt_train_gbm[target_gbm] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid_gbm[target_gbm] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_gbm = melt_train_gbm.dropna()

    melt_train_gbm['sales_next_month'] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(-1)

    melt_valid_gbm['sales_next_month'] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_gbm = melt_train_gbm.dropna()

    melt_train_gbm["lag_sales_1"] = melt_train_gbm.groupby("Product_Code")['Sales'].shift(1)

    melt_valid_gbm["lag_sales_1"] = melt_valid_gbm.groupby("Product_Code")['Sales'].shift(1)

    melt_train_gbm["diff_sales_1"] = melt_train_gbm.groupby("Product_Code")['Sales'].diff(1)

    melt_valid_gbm["diff_sales_1"] = melt_valid_gbm.groupby("Product_Code")['Sales'].diff(1)

    melt_train_gbm["mean_sales_4"] = melt_train_gbm.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    melt_valid_gbm["mean_sales_4"] = melt_valid_gbm.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    features_gbm = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
    target_gbm = 'sales_next_month'

    X_train_gbm = melt_train_gbm[features_gbm]
    y_train_gbm = melt_train_gbm[target_gbm]
    X_valid_gbm = melt_valid_gbm[features_gbm]
    y_valid_gbm = melt_valid_gbm[target_gbm]
    
    imputer = SimpleImputer()
    X_train_imputed_gbm = imputer.fit_transform(X_train_gbm)
    X_valid_imputed_gbm = imputer.transform(X_valid_gbm)

    gbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=0, n_jobs=6)
    gbm_model.fit(X_train_imputed_gbm, y_train_gbm)

    y_pred_gbm = gbm_model.predict(X_valid_imputed_gbm)

    mape_gbm = mape(y_valid_gbm, y_pred_gbm)
    wmape_gbm = wmape(y_valid_gbm, y_pred_gbm)

    print("LightGBM MAPE:", mape_gbm)
    print("LightGBM WMAPE:", wmape_gbm)

    new_examples_gbm = melt_valid_gbm[melt_valid_gbm['Month'] == 81].copy()
    new_examples_imputed_gbm = pd.DataFrame(imputer.fit_transform(new_examples_gbm[features_gbm]), columns=features_gbm)

    predictions_gbm_future = gbm_model.predict(new_examples_imputed_gbm)

    new_examples_gbm['p_sales_next_month_gbm'] = predictions_gbm_future

    print(new_examples_gbm[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_gbm']])

# Örnek Kullanım:
train_and_evaluate_gbm_model(df_week)

#%%

import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def train_and_evaluate_xgb_model(df_week):
    melt_xgb = df_week.copy()

    melt_xgb['Product_Code'] = df_week['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
    melt_xgb['Month'] = df_week['Ay']
    melt_xgb['Week'] = df_week['Hafta']
    melt_xgb['Sales'] = melt_xgb['Miktar']

    melt_xgb = melt_xgb.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

    melt_xgb['Product_Code'] = label_encoder.fit_transform(melt_xgb['Product_Code']) + 1

    split_point_xgb = 65
    melt_train_xgb = melt_xgb[melt_xgb['Month'] < split_point_xgb].copy()
    melt_valid_xgb = melt_xgb[melt_xgb['Month'] >= split_point_xgb].copy()

    target_xgb = 'sales_next_month'

    melt_train_xgb[target_xgb] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid_xgb[target_xgb] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_xgb = melt_train_xgb.dropna()

    melt_train_xgb['sales_next_month'] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(-1)

    melt_valid_xgb['sales_next_month'] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(-1)

    melt_train_xgb = melt_train_xgb.dropna()

    melt_train_xgb["lag_sales_1"] = melt_train_xgb.groupby("Product_Code")['Sales'].shift(1)

    melt_valid_xgb["lag_sales_1"] = melt_valid_xgb.groupby("Product_Code")['Sales'].shift(1)

    melt_train_xgb["diff_sales_1"] = melt_train_xgb.groupby("Product_Code")['Sales'].diff(1)

    melt_valid_xgb["diff_sales_1"] = melt_valid_xgb.groupby("Product_Code")['Sales'].diff(1)

    melt_train_xgb["mean_sales_4"] = melt_train_xgb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    melt_valid_xgb["mean_sales_4"] = melt_valid_xgb.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    features_xgb = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']

    X_train_xgb = melt_train_xgb[features_xgb]
    y_train_xgb = melt_train_xgb[target_xgb]
    X_valid_xgb = melt_valid_xgb[features_xgb]
    y_valid_xgb = melt_valid_xgb[target_xgb]
    
    imputer = SimpleImputer()
    X_train_imputed_xgb = imputer.fit_transform(X_train_xgb)
    X_valid_imputed_xgb = imputer.transform(X_valid_xgb)

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=0, n_jobs=6)
    xgb_model.fit(X_train_imputed_xgb, y_train_xgb)

    y_pred_xgb = xgb_model.predict(X_valid_imputed_xgb)

    mape_xgb = mape(y_valid_xgb, y_pred_xgb)
    wmape_xgb = wmape(y_valid_xgb, y_pred_xgb)

    print("XGBoost MAPE:", mape_xgb)
    print("XGBoost WMAPE:", wmape_xgb)

    new_examples_xgb = melt_valid_xgb[melt_valid_xgb['Month'] == 81].copy()
    new_examples_imputed_xgb = pd.DataFrame(imputer.fit_transform(new_examples_xgb[features_xgb]), columns=features_xgb)

    predictions_xgb_future = xgb_model.predict(new_examples_imputed_xgb)

    new_examples_xgb['p_sales_next_month_xgb'] = predictions_xgb_future

    print(new_examples_xgb[['Product_Code', 'Month', 'Sales', 'p_sales_next_month_xgb']])

# Örnek Kullanım:
train_and_evaluate_xgb_model(df_week)


#%%
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import catboost
from catboost import CatBoostRegressor


df = pd.read_excel('new_filtered.xlsx')

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k', linewidth=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmini Değerler')
    plt.title(f'{model_name} - Gerçek vs. Tahmini Değerler')
    plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

k = 3

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=k,
                random_state=1))
])


def plot_feature_importance(model, features, model_name):
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
    feature_importance = pd.Series(importances, index=features)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.xlabel('Özellik Önem Derecesi')
    plt.title(f'{model_name} - Importance')
    plt.show()

def plot_error_distribution(y_true, y_pred, model_name):
    errors = y_true - y_pred
    plt.hist(errors, bins=30)
    plt.xlabel('Hata (Gerçek - Tahmini)')
    plt.ylabel('Frekans')
    plt.title(f'{model_name} - Hata Dağılımı')
    plt.show()
    
def train_and_evaluate_model(df):
    
    def mape(y_true, y_pred):
        ape = np.abs((y_true - y_pred) / y_true)
        ape[~np.isfinite(ape)] = 1.  # pessimist estimate
        return np.mean(ape)

    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    label_encoder = LabelEncoder()
    imputer = SimpleImputer()    
    
    melt = df.copy()
    
    melt['Ay'] = (melt['Tarih'].dt.year - melt['Tarih'].dt.year.min()) * 12 + melt['Tarih'].dt.month
    melt['Hafta'] = (melt['Tarih'] - melt['Tarih'].min()).dt.days // 7 + 1
    

    melt['Month'] = melt['Ay']
    melt['Week'] = melt['Hafta']
    # melt['Customer_Code'] = melt['Müşteri Kodu']
    melt['Sales'] = melt['Miktar']
    melt['Product_Label'] = melt['Ürün Kodu']
    melt['Product_Code'] = melt['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
    melt = melt.dropna()
    # melt['Customer_Code'] = melt['Customer_Code'].str.extract('(\d+)', expand=False).astype(int)
    
    print(melt.head())
    print(melt.head(-5))
    
    melt = melt.drop(['Ürün Kodu', 'Müşteri Kodu', 'Net Tutar', 'Toplam', 'Hafta', 'Ay', 'Birim Fiyat', 'Tarih', 'Yıl', 'Miktar'], axis=1)

    melt['Product_Code'] = label_encoder.fit_transform(melt['Product_Code']) + 1
    # melt['Customer_Code'] = label_encoder.fit_transform(melt['Customer_Code']) + 1
    
    split_point = 65
    melt_train = melt[melt['Month'] < split_point].copy()
    melt_valid = melt[melt['Month'] >= split_point].copy()

    target = 'sales_next_month'

    melt_train[target] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid[target] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)

    melt_train = melt_train.dropna()

    melt_train['sales_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
    melt_valid['sales_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)
    melt_train = melt_train.dropna()

    melt_train["lag_sales_1"] = melt_train.groupby("Product_Code")['Sales'].shift(1)
    melt_valid["lag_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].shift(1)

    melt_train["diff_sales_1"] = melt_train.groupby("Product_Code")['Sales'].diff(1)
    melt_valid["diff_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].diff(1)

    melt_train["mean_sales_4"] = melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)
    melt_valid["mean_sales_4"] = melt_valid.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

    features = [ 'Product_Code', 'Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4','Month','Week']
    target = 'sales_next_month'

    X_train = melt_train[features]
    y_train = melt_train[target]
    X_valid = melt_valid[features]
    y_valid = melt_valid[target]

    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
    linear_reg_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)

    models = [
        (lgb.LGBMRegressor(n_estimators=100, random_state=0, n_jobs=6), "LGBM", 'lgb',
         {'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 30, 50]}),
        
        (xgb.XGBRegressor(n_estimators=100, random_state=0, n_jobs=6), "XGBoost",'xgb',
        {'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5]}),
        (GradientBoostingRegressor(n_estimators=100, random_state=0),"GradientBoost","gb_model",
         {'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]}),
        # (LinearRegression(),"LinearReg","linear_reg_model"),
        (RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6), "RandomForest", "rf_model",
         {'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}),
        # (catboost.CatBoostRegressor(iterations=100, random_state=0, task_type='GPU', devices='0:1'), "CatBoost", 'catboost',
        #  {'iterations': [50, 100, 200],
        #   'learning_rate': [0.01, 0.1, 0.2],
        #   'depth': [3, 5, 7]},  # grid_params eklendi
        #  )
        ]

    for model,name,model_type,grid_params in models:
        
        delfault_params = model.get_params()
        print("Default parameters: "+str(delfault_params))
        
        if name == "LinearReg":
            model.fit(X_train_imputed, y_train)
        else:
            grid_search = GridSearchCV(model, grid_params, scoring='neg_mean_absolute_error', cv=3)
            grid_search.fit(X_train_imputed, y_train)
            best_params = grid_search.best_params_
            model.fit(X_train_imputed, y_train)
            
            model = model.set_params(**best_params)
            model.fit(X_train_imputed, y_train)   
        
        y_pred = model.predict(X_valid_imputed)
        
        print("Params after grid search: "+str(best_params))
        mape_val = mape(y_valid, y_pred)
        wmape_val = wmape(y_valid, y_pred)
    
        print(f"{model_type.upper()} MAPE: {mape_val}")
        print(f"{model_type.upper()} WMAPE: {wmape_val}")
     
        new_examples = melt_valid[melt_valid['Month'] == 81].copy()
        new_examples_imputed = pd.DataFrame(imputer.fit_transform(new_examples[features]), columns=features)
    
        predictions_future = model.predict(new_examples_imputed)
    
        new_examples[f'p_sales_next_month_{name}'] = predictions_future
    
        print(new_examples[['Product_Label','Product_Code', 'Month', 'Sales', f'p_sales_next_month_{name}']])
        
        plot_feature_importance(model, features, name)
        plot_error_distribution(y_valid, y_pred, name)
        plot_actual_vs_predicted(y_valid, y_pred, name)

    print(melt_valid['Product_Code'].nunique())

train_and_evaluate_model(df)

"""
ürün adını da df'e ekle hangi ürün olduğu belli olsun
lineer trend oluştur
training 
"""

#%%
import pandas as pd
from prophet import Prophet

df = pd.read_excel('new_filtered.xlsx')

df_train = df.copy()

print(df_train.info())

# df_train['Müşteri Kodu'] = df_train['Müşteri Kodu'].astype('category').cat.codes

df_train.rename(columns={'Tarih': 'ds', 'Miktar': 'y'}, inplace = True)


#%% PROPHET

model = Prophet()

model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 'Müşteri Kodu' ve 'Birim Fiyat' sütunlarını regresör olarak ekleyin
# model.add_regressor('Müşteri Kodu')
# model.add_regressor('Birim Fiyat')


model.fit(df_train)

future = model.make_future_dataframe(periods=365)
future.tail()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = model.plot(forecast)

fig2 = model.plot_components(forecast)

#%%

print(df_train.head())

print(df_filtered.head())

