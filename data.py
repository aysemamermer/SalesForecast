import pandas as pd
import matplotlib.pyplot as plt

# Verileri okur
data = pd.read_excel('data_new.xlsx')
# Tüm null değerlerin olduğu satırlar kaldır
data = data.dropna(how='all')

# Veri tiplerini dönüştür
data['Tarih'] = pd.to_datetime(data['Tarih'])
data['Yıl'] = data['Yıl'].astype('object')
data['İrsaliye No'] = data['İrsaliye No'].astype('object')
data['Müşteri Kodu'] = data['Müşteri Kodu'].astype('object')
data['Sevk Müşterisi'] = data['Sevk Müşterisi'].astype('object')
data['Ürün Kodu'] = data['Ürün Kodu'].astype('object')
data['Miktar'] = data['Miktar'].astype('float')
data['Birim Fiyat'] = data['Birim Fiyat'].astype('float')
data['Toplam'] = data['Toplam'].astype('float')
data['Net Tutar'] = data['Net Tutar'].astype('float')

# İki farklı çizim oluştur
plt.plot(data["Tarih"], data['Toplam'])
plt.xlabel("Tarih")
plt.ylabel("Satış Miktarı (Tutar)")
plt.show()

plt.plot(data["Tarih"], data['Miktar'])
plt.xlabel("Tarih")
plt.ylabel("Satış Miktarı(Adet)")
plt.show()

# Veri tiplerini görüntüle
print(data.dtypes)
