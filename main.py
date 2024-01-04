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

