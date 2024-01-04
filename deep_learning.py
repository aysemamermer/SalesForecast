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

df = pd.read_excel('new_filtered.xlsx')

# df['Tarih'] = pd.to_datetime(df['Tarih'])

df.info()
df = df.dropna()
print("asdas1",df.head())
print(df.columns)

df_month = df.copy()

df_month['Tarih'] = pd.to_datetime(df_month['Tarih'])

# Tarihlere göre sıralama yapın
df_month = df_month.sort_values('Tarih')

# Hafta numaralarını hesaplayın
df_month['Ay'] = (df_month['Tarih'].dt.year - df_month['Tarih'].dt.year.min()) * 12 + df_month['Tarih'].dt.month

df_month = df_month.drop(['Yıl', 'Müşteri Kodu', 'Toplam', 'Net Tutar'], axis = 1)

print("asdas2",df_month.head())
#%%
melt = df_month.copy()

melt['Product_Code'] = df_month['Ürün Kodu'].str.extract('(\d+)', expand=False).astype(int)
melt['Month'] = df_month['Ay']
melt['Sales'] = df_month['Miktar']
melt['Price'] = df_month['Birim Fiyat']
melt['Date'] = df_month['Tarih']

melt = melt.drop(['Ürün Kodu','Ay', 'Birim Fiyat', 'Tarih', 'Miktar'], axis = 1)
print("asdas3",melt)
dataset = melt.pivot_table(index = ['Product_Code'],values = ['Sales'],columns = ['Month'],fill_value = 0,aggfunc='sum')
print(dataset.head(3))


melt = melt.sort_values(['Month', 'Product_Code'])
product_num = melt['Product_Code'].nunique()
print("asdas4","Farklı Ürün Sayısı", product_num)

print(melt.info())
label_encoder = LabelEncoder()
melt['Product_Code'] = label_encoder.fit_transform(melt['Product_Code']) + 1  # +1 ekleyerek 0'dan değil, 1'den başlatın
print("asdas5",melt.head())


pro_selected = melt.groupby("Product_Code")["Sales"].sum().sort_values(ascending=False).index[0]
df_pro_selected = melt[melt["Product_Code"] == pro_selected]


df_pro_selected= df_pro_selected.sort_values(by=['Product_Code'], ascending=True)
print("asdasson",df_pro_selected.info())


#%% LSTM v1

from sklearn.metrics import mean_squared_error
from math import sqrt

melt['Year'] = melt['Date'].dt.year
melt = melt.drop(['Date'], axis=1)


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

# LSTM modelini oluşturun
model = Sequential()
model.add(LSTM(units=20, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitin
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Modelin performansını değerlendirin
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))

y_pred = model.predict(X_test)


try:
    # Eğitim ve doğrulama (validation) kayıplarını görselleştirin
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

except: 
    pass
        # Tahminleri ve gerçek değerleri görselleştirin


try:
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred)
    
    plt.plot(y_test_original, label='Actual Sales')
    plt.plot(y_pred_original, label='Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
except:
    pass
print("LSTM v1")
print('Test RMSE:', rmse)

#%% LSTM v2

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

label_encoder = LabelEncoder()


df = pd.read_excel('new_filtered.xlsx')

df_week = df.copy()
df_week['Tarih'] = pd.to_datetime(df_week['Tarih'])

# Tarihlere göre sıralama yapın
df_week = df_week.sort_values('Tarih')

# Hafta numaralarını hesaplayın
df_week['Hafta'] = (df_week['Tarih'] - df_week['Tarih'].min()).dt.days // 7 + 1
df_week['Ay'] = (df_week['Tarih'].dt.year - df_week['Tarih'].dt.year.min()) * 12 + df_week['Tarih'].dt.month

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

try:
    # Eğitim ve doğrulama (validation) kayıplarını görselleştirin
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


except:
    pass

try:

    # Eğitim ve doğrulama (validation) MAE'yi görselleştirin
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
except:
    pass


try:
    # Gerçek ve tahmin edilen değerleri görselleştirin
    plt.plot(validY, label='Actual Sales')
    plt.plot(validPredict, label='Predicted Sales')
    plt.title('Actual vs Predicted Sales (Validation Set)')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

except:
    pass

try:
    # Gelecekteki tahminleri görselleştirin
    plt.plot(new_examples_lstm['Sales'], label='Actual Sales')
    plt.plot(new_examples_lstm['p_sales_next_month_lstm'], label='Predicted Sales (Future)')
    plt.title('Actual vs Predicted Sales (Future Predictions)')
    plt.xlabel('Example Index')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
except:
    pass


#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import pandas as pd
df = pd.read_excel("new_filtered.xlsx")

df = df[5::6]

print(df.info())

df = df.drop(['Müşteri Kodu', 'Net Tutar', 'Toplam', 'Birim Fiyat', 'Yıl'], axis=1)


df.index = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
df[:26]


temp = df['Miktar']
temp.plot()


import numpy as np

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape


X_train1, y_train1 = X1[:22000], y1[:22000]
X_val1, y_val1 = X1[22000:25000], y1[22000:25000]
X_test1, y_test1 = X1[25000:], y1[25000:]
print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)



model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()


cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])

from tensorflow.keras.models import load_model
model1 = load_model('model1/')

train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})


import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])



import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])

plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])



test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results


plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])


from sklearn.metrics import mean_squared_error as mse

def plot_predictions1(model, X, y, start=0, end=100):
  predictions = model.predict(X).flatten()
  df = pd.DataFrame(data={'Predictions': predictions, 'Actuals':y})
  plt.plot(df['Predictions'][start:end])
  plt.plot(df['Actuals'][start:end])
  return df, mse(predictions, y)

plot_predictions1(model1, X_test1, y_test1)


model2 = Sequential()
model2.add(InputLayer((5, 1)))
model2.add(Conv1D(64, kernel_size=2, activation='relu'))
model2.add(Flatten())
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))
model2.summary()

cp2 = ModelCheckpoint('model2/', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


model2.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp2])


plot_predictions1(model2, X_test1, y_test1)



