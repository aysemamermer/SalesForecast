import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns

#%%------------------------------Read data------------------------------

tqdm.pandas()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_excel('data_new.xlsx')
print("Training Data Shape:", data.shape)

# İlk 5 satırı görüntüle
print(data.head())

# Veri seti hakkında temel istatistikleri görüntüle
print(data.describe())

# Veri setinin sütunlarını ve veri tiplerini incele
print(data.info())

data_clean = data.drop(['Sevk Müşterisi', 'İrsaliye No', 'SPECODE5'], axis=1)

data_clean.dropna()

# Kaldırılmış veriyi gözden geçir
print("cleaned data", data_clean.head(5))

data_clean.to_csv('cleaned_data.csv', index=False)

# Tarih sütununu datetime nesnelerine dönüştür
data_clean['Tarih'] = pd.to_datetime(data_clean['Tarih'], format='%Y-%m-%d %H:%M:%S.%f')
data_clean['Yıl'] = data_clean['Yıl'].astype('int32')
data_clean['Miktar'] = data_clean['Miktar'].astype('float')
data_clean['Birim Fiyat'] = data_clean['Birim Fiyat'].astype('float')
data_clean['Toplam'] = data_clean['Toplam'].astype('float')
data_clean['Net Tutar'] = data_clean['Net Tutar'].astype('float')

print(data.dtypes)

# Tarih sütununu kullanarak görselleştirme yapabilirsiniz
plt.figure(figsize=(10, 6))
plt.plot(data_clean['Tarih'], data_clean['Miktar'])
plt.xlabel('Tarih')
plt.ylabel('Miktar')
plt.title('Satış Miktarı Zaman İçinde')
plt.xticks(rotation=45)
plt.show()

# Ürünlerin satışlarının dağılımını gösteren bir kutu grafiği
sns.boxplot(x='Ürün Kodu', y='Miktar', data=data_clean)
plt.title('Ürün Kodlarına Göre Satışlar')
plt.xlabel('Ürün Kodu')
plt.ylabel('Satış Miktarı')
plt.xticks(rotation=45)
plt.show()

print("Training Data Shape:", data_clean.shape)

# Müşterilere göre toplam satış miktarı
müşteri_satışları = data_clean.groupby('Müşteri Kodu')['Miktar'].sum().reset_index()
print(müşteri_satışları.sort_values(by='Miktar', ascending=False))

# Ürün kodlarına göre toplam satış miktarı
ürün_satışları = data_clean.groupby('Ürün Kodu')['Miktar'].sum().reset_index()
print(ürün_satışları.sort_values(by='Miktar', ascending=False))

