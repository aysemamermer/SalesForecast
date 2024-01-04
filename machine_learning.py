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
import pickle
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


"""
günlüğe çevir

"""

RunID = datetime.now().strftime("%Y%m%d-%H%M")

k = 3

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=k,
                random_state=1))
])


def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    #ape[~np.isfinite(ape)] = 0. # VERY questionable
    ape[~np.isfinite(ape)] = 1. # pessimist estimate
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k', linewidth=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmini Değerler')
    plt.title(f'{model_name} - Gerçek vs. Tahmini Değerler')
    plt.show()


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

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    
def train_and_evaluate_model(df):
    
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
        # (lgb.LGBMRegressor(n_estimators=100, random_state=0, n_jobs=6), "LGBM", 'lgb',
        #  {'n_estimators': [50, 100, 200],
        # 'learning_rate': [0.01, 0.1, 0.2],
        # 'num_leaves': [31, 50, 100],
        # 'min_child_samples': [20, 30, 50]}),
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
        ]

    for model,name,model_type,grid_params in models:
        
        save_model(model, str(name + RunID))
        
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


if __name__ == '__main__':
    # Veriyi yükle
    df = pd.read_excel('new_filtered.xlsx')
    train_and_evaluate_model(df)

"""
ürün adını da df'e ekle hangi ürün olduğu belli olsun
lineer trend oluştur
training 
"""


#%% model deneme

# def load_model(filename):
#     with open(filename, 'rb') as file:
#         loaded_model = pickle.load(file)
#     return loaded_model



# target = 'sales_next_month'

# melt_train = pd.DataFrame()

# melt_train[target] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
# melt_valid[target] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)

# melt_train = melt_train.dropna()

# melt_train['sales_next_month'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
# melt_valid['sales_next_month'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)
# melt_train = melt_train.dropna()

# melt_train["lag_sales_1"] = melt_train.groupby("Product_Code")['Sales'].shift(1)
# melt_valid["lag_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].shift(1)

# melt_train["diff_sales_1"] = melt_train.groupby("Product_Code")['Sales'].diff(1)
# melt_valid["diff_sales_1"] = melt_valid.groupby("Product_Code")['Sales'].diff(1)

# melt_train["mean_sales_4"] = melt_train.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)
# melt_valid["mean_sales_4"] = melt_valid.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

# features = [ 'Product_Code', 'Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4','Month','Week']
# target = 'sales_next_month'

# new_data = pd.DataFrame({
#     'Product_Code': ["your_product_code_value"],
#     'lag_sales_1': [your_lag_sales_1_value],
#     'diff_sales_1': [your_diff_sales_1_value],
#     'mean_sales_4': [your_mean_sales_4_value],
#     'Month': [your_month_value],
#     'Week': [your_week_value],
#     'Sales': [your_sales_value],
# })














#%% ARIMA

df_filtered = df.copy()


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



#%% ARIMA en çok satan ürün için

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


#%%

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def plot_stl_ets_forecast(df, target_column, period=12, forecast_steps=12):
    # STL ayrıştırma
    result_stl = seasonal_decompose(df[target_column], model='additive', period=period)
    df_stl = pd.DataFrame({
        'Trend': result_stl.trend,
        'Seasonal': result_stl.seasonal,
        'Residual': result_stl.resid
    })

    # ETS modelini eğit
    ets_model = ExponentialSmoothing(df[target_column], trend='add', seasonal='add', seasonal_periods=period)
    ets_fit = ets_model.fit()

    # Tahmin yap
    forecast_ets = ets_fit.forecast(steps=forecast_steps)

    # Tahmin ve gerçek değerleri karşılaştır
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(df[target_column], label='Gerçek Veri')
    plt.title('Gerçek Veri')

    plt.subplot(3, 1, 2)
    plt.plot(df_stl['Trend'], label='Trend')
    plt.plot(df_stl['Seasonal'], label='Seasonal')
    plt.plot(df_stl['Residual'], label='Residual')
    plt.title('STL Ayrıştırma')

    plt.subplot(3, 1, 3)
    plt.plot(df[target_column], label='Gerçek Veri')
    plt.plot(forecast_ets.index, forecast_ets, label='ETS Tahmini', linestyle='dashed')
    plt.title('ETS Tahmini')

    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Veriyi yükle
    df = pd.read_excel('new_filtered.xlsx')

    # Tarih sütununu indeks olarak ayarla
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df.set_index('Tarih', inplace=True)

    # Fonksiyonu çağır
    plot_stl_ets_forecast(df, target_column='Miktar', period=12, forecast_steps=12)

    
#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle

def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    ape[~np.isfinite(ape)] = 1.0  # Pessimist estimate for non-finite values
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k', linewidth=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmini Değerler')
    plt.title(f'{model_name} - Gerçek vs. Tahmini Değerler')
    plt.show()

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

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def predict_sales(model, date, product_code, imputer, label_encoder):
    # Tarih sütununu ayrıştırma
    date_parts = pd.to_datetime(date)
    month = date_parts.month
    week = (date_parts - df['Tarih'].min()).days // 7 + 1

    # Tahmin yapmak için giriş verisini oluştur
    input_data = {
        'Product_Code': label_encoder.transform([product_code])[0] + 1,
        'Month': month,
        'Week': week
        # Diğer özelliklerinizi buraya ekleyebilirsiniz
    }

    # Boş değerleri doldur
    input_df = pd.DataFrame(input_data, index=[0])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Tahmin yap
    prediction = model.predict(input_df_imputed)[0]
    return prediction

def train_and_evaluate_model(df):
    
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
        # (lgb.LGBMRegressor(n_estimators=100, random_state=0, n_jobs=6), "LGBM", 'lgb',
        #  {'n_estimators': [50, 100, 200],
        # 'learning_rate': [0.01, 0.1, 0.2],
        # 'num_leaves': [31, 50, 100],
        # 'min_child_samples': [20, 30, 50]}),
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
        ]

    for model,name,model_type,grid_params in models:
        
        # save_model(model, str(name + RunID))
        
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

if __name__ == '__main__':
    # Veriyi yükle
    df = pd.read_excel('new_filtered.xlsx')

    # Tarih sütununu ayrıştırma ve modele ekleme
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day

    # Modeli eğit ve değerlendir
    train_and_evaluate_model(df)

    # Örnek bir tahmin yapma
    target_date = '2024-01-01'
    product_code = 'Y1449'
    label_encoder = LabelEncoder()
    imputer = SimpleImputer()  
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
    
    prediction = predict_sales(gb_model, target_date, product_code, imputer, label_encoder)
    print(f"{target_date} tarihinde {product_code} ürün kodu için tahmini satış: {prediction}")

#%% doğru çalışıyor XGB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import pickle

def prepare_data(df):
    # Tarih sütununu ayrıştırma
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day

    # Kategorik özellikleri sayısallaştırma
    label_encoder = LabelEncoder()
    df['Product_Code'] = label_encoder.fit_transform(df['Ürün Kodu'])

    # Eğitim verisi ve hedef değişkeni seçme
    features = ['Year', 'Month', 'Day', 'Product_Code']
    target = 'Miktar'

    X = df[features]
    y = df[target]

    # Eksik değerleri doldurma
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standartlaştırma
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, label_encoder, imputer, scaler

def train_xgb_model(X, y):
    # XGBoost modelini eğitme
    model = xgb.XGBRegressor(n_estimators=100, random_state=0)
    """
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
    linear_reg_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)

    """
    
    model.fit(X, y)
    return model

def save_model(model, filename):
    # Modeli pickle formatında kaydetme
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def predict_sales(model, date, product_code, label_encoder, imputer, scaler):
    # Tahmin yapmak için giriş verisini oluşturma
    date_parts = pd.to_datetime(date)
    input_data = pd.DataFrame({
        'Year': date_parts.year,
        'Month': date_parts.month,
        'Day': date_parts.day,
        'Product_Code': label_encoder.transform([product_code])[0]
    }, index=[0])

    # Boş değerleri doldur, standartlaştırma uygula
    input_data_imputed = imputer.transform(input_data)
    input_data_scaled = scaler.transform(input_data_imputed)

    # Tahmin yap
    prediction = model.predict(input_data_scaled)[0]
    return prediction

if __name__ == '__main__':
    # Veriyi yükle
    

    df = pd.read_excel('new_filtered.xlsx')

    # Veriyi hazırla
    X, y, label_encoder, imputer, scaler = prepare_data(df)

    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Modeli eğit
    model = train_xgb_model(X_train, y_train)


    # Modeli kaydet
    save_model(model, 'xgb_model.pkl')
#%%
    # Örnek bir tahmin yap
    target_date = '2024-03-02'
    product_code = 'Y1449'
    prediction = predict_sales(model, target_date, product_code, label_encoder, imputer, scaler)
    print(f"{target_date} tarihinde {product_code} ürün kodu için tahmini satış: {prediction}")


#%% 3



from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

RunID = datetime.now().strftime("%Y%m%d-%H%M")

def prepare_data(df):
    # Tarih sütununu ayrıştırma
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day

    # Kategorik özellikleri sayısallaştırma
    label_encoder = LabelEncoder()
    df['Product_Code'] = label_encoder.fit_transform(df['Ürün Kodu'])

    # Eğitim verisi ve hedef değişkeni seçme
    features = ['Year', 'Month', 'Day', 'Product_Code']
    y = 'Miktar'


    # Eksik değerleri doldurma
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[features])

    # Standartlaştırma
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, df[y], label_encoder, imputer, scaler

def train_and_save_models(X_train, y_train, model_filename):
    # XGBoost modelini eğitme
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=0)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
    linear_reg_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
    
    models = [
        (xgb_model, "xgb_model"),
        (gb_model, "gb_model"),
        (linear_reg_model, "lr_model"),
        (rf_model, "rf_model")
    ]

    trained_models = []

    for model, name in models:
        print("Training model: {}".format(name))
        model.fit(X_train, y_train)

        # Modeli kaydetme
        filename = "{}{}_{}.pkl".format(name, model_filename, RunID)
        with open(filename, 'wb') as file:
            pickle.dump((model, label_encoder, imputer, scaler), file)
        
        trained_models.append(filename)
        print("Model saved: {}".format(name))

    return trained_models

def load_model(filename):
    # Kaydedilmiş modeli yükleme
    with open(filename, 'rb') as file:
        return pickle.load(file)

def predict_sales(model, target_date, product_code, label_encoder, imputer, scaler, df):
    target_date = pd.to_datetime(target_date)
    product_code_encoded = label_encoder.transform([product_code])[0]

    # Belirtilen tarih ve ürün koduna sahip bir satırın olup olmadığını kontrol et
    filtered_row = df[(df['Tarih'] == target_date) & (df['Product_Code'] == product_code_encoded)]


    # Yeni özellikleri ekleyerek input_data'yı güncelle
    input_data = np.array([
        [target_date.year, target_date.month, target_date.day, product_code_encoded]
    ])

    # Eksik değerleri doldur
    input_data_imputed = imputer.transform(input_data)

    # Standartlaştır
    input_data_scaled = scaler.transform(input_data_imputed)

    model, _, _, _ = model

    # Tahmin yap
    prediction = model.predict(input_data_scaled)

    return prediction[0]

if __name__ == '__main__':
    # Veriyi yükle
    df = pd.read_excel('new_filtered.xlsx')  # Güncellenmiş veri seti

    # Veriyi hazırla
    X, y, label_encoder, imputer, scaler = prepare_data(df)

    # Veriyi eğitim ve test setlerine ayır
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    # Modeli eğit ve kaydet
    loaded_models = train_and_save_models(X_train, y_train, 'model')

    # Gelecekteki bir tarih için tahmin yap
    for model_name in loaded_models:
        loaded_model = load_model(model_name)

        # Örnek bir tahmin yap
        target_date = '2025-01-01'
        product_code = 'Y1449'
        
        prediction = predict_sales(loaded_model, target_date, product_code, label_encoder, imputer, scaler, df)
        print(f"{target_date} tarihinde {product_code} ürün kodu için tahmini satış: {prediction}")
#%% 2

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

RunID = datetime.now().strftime("%Y%m%d-%H%M")

def prepare_data(df):
    # Tarih sütununu ayrıştırma
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day

    # Kategorik özellikleri sayısallaştırma
    label_encoder = LabelEncoder()
    df['Product_Code'] = label_encoder.fit_transform(df['Ürün Kodu'])

    # Eğitim verisi ve hedef değişkeni seçme
    features = ['Year', 'Month', 'Day', 'Product_Code', 'diff_sales_1']
    y = 'Miktar'

    # Yeni değişkenleri ekle
    df['diff_sales_1'] = df.groupby("Product_Code")['Miktar'].diff(1)

    # Eksik değerleri doldurma
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[features])

    # Standartlaştırma
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, df[y], label_encoder, imputer, scaler

def train_and_save_models(X_train, y_train, model_filename):
    # RandomForestRegressor modelini eğitme
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
    
    models = [
        (rf_model, "rf_model")
    ]

    trained_models = []

    for model, name in models:
        print("Training model: {}".format(name))
        model.fit(X_train, y_train)

        # Modeli kaydetme
        filename = "{}{}_{}.pkl".format(name, model_filename, RunID)
        with open(filename, 'wb') as file:
            pickle.dump((model, label_encoder, imputer, scaler), file)
        
        trained_models.append(filename)
        print("Model saved: {}".format(name))

    return trained_models

def load_model(filename):
    # Kaydedilmiş modeli yükleme
    with open(filename, 'rb') as file:
        return pickle.load(file)

def predict_sales(model, target_date, product_code, label_encoder, imputer, scaler, df):
    target_date = pd.to_datetime(target_date)
    product_code_encoded = label_encoder.transform([product_code])[0]

    # Belirtilen tarih ve ürün koduna sahip bir satırın olup olmadığını kontrol et
    filtered_row = df[(df['Tarih'] == target_date) & (df['Product_Code'] == product_code_encoded)]

    # Gerekli özellikleri çek
    diff_sales_1 = filtered_row['diff_sales_1'].values[0]

    # Yeni özellikleri ekleyerek input_data'yı güncelle
    input_data = np.array([
        [target_date.year, target_date.month, target_date.day, product_code_encoded, diff_sales_1]
    ])

    # Eksik değerleri doldur
    input_data_imputed = imputer.transform(input_data)

    # Standartlaştır
    input_data_scaled = scaler.transform(input_data_imputed)

    model, _, _, _ = model

    # Tahmin yap
    prediction = model.predict(input_data_scaled)

    return prediction[0]

if __name__ == '__main__':
    # Veriyi yükle
    df = pd.read_excel('new_filtered.xlsx')

    # Veriyi hazırla
    X, y, label_encoder, imputer, scaler = prepare_data(df)

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Modeli eğit ve kaydet
    loaded_models = train_and_save_models(X_train, y_train, 'model')
    
#%%
    # Tahmin yap
    for model_name in loaded_models:
        loaded_model = load_model(model_name)

        # Örnek bir tahmin yap
        target_date = '2025-04-01'
        product_code = 'Y1449'
        
        prediction = predict_sales(loaded_model, target_date, product_code, label_encoder, imputer, scaler, df)
        print(f"{target_date} tarihinde {product_code} ürün kodu için tahmini satış: {prediction}")

#%%

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pickle
from datetime import datetime
# Veriyi yükle


m_df= pd.read_excel('new_filtered.xlsx')
# Veriyi tarih sütununa göre sırala
#%%
df = m_df.copy()

# Sadece belirli sütunları seç
# Sadece belirli sütunları seç
#%%
df = df.drop(['Müşteri Kodu', 'Net Tutar', 'Toplam', 'Birim Fiyat', 'Yıl'], axis=1)

# df = df[['Tarih', 'Ürün Kodu', 'Miktar']]

df['Tarih'] = pd.to_datetime(df['Tarih'])
df.set_index('Tarih', inplace=True)

# Veriyi eğitim ve test setlerine ayır
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# SARIMA Modelini Eğitme
sarima_order = (1, 1, 1, 12)  # Örnek order
sarima_model = SARIMAX(train['Miktar'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()

# Modeli Kaydetme
sarima_model_filename = "sarima_model.pkl"
with open(sarima_model_filename, 'wb') as file:
    pickle.dump(sarima_model_fit, file)

# ARIMA Modelini Eğitme
arima_order = (5, 1, 0)  # Örnek order
arima_model = ARIMA(train['Miktar'], order=arima_order)
arima_model_fit = arima_model.fit()

# Modeli Kaydetme
arima_model_filename = "arima_model.pkl"
with open(arima_model_filename, 'wb') as file:
    pickle.dump(arima_model_fit, file)

# SARIMA Modelini Kullanarak Tahmin Yapma
def predict_sarima(model, start_date, end_date, product_code):
    forecast = model.get_forecast(steps=(end_date - start_date).days)
    pred_values = forecast.predicted_mean
    return pred_values[start_date:end_date]

# ARIMA Modelini Kullanarak Tahmin Yapma
def predict_arima(model, start_date, end_date, product_code):
    forecast = model.get_forecast(steps=(end_date - start_date).days)
    pred_values = forecast.predicted_mean
    return pred_values[start_date:end_date]

# SARIMA Modelini Kullanarak Tahmin Yapma
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 10)
product_code = 'ABC123'

sarima_prediction = predict_sarima(sarima_model_fit, start_date, end_date, product_code)
print("SARIMA Tahminleri:")
print(sarima_prediction)

# ARIMA Modelini Kullanarak Tahmin Yapma
arima_prediction = predict_arima(arima_model_fit, start_date, end_date, product_code)
print("ARIMA Tahminleri:")
print(arima_prediction)