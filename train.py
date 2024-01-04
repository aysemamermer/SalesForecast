#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import numpy as np
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RunID = datetime.now().strftime("%Y%m%d-%H%M")

pd.set_option("display.max_columns", None)

# Veriyi hazırla
file_path = "new_filtered.xlsx"
df = pd.read_excel(file_path)



#%%

model_file = "C:/Users/aysem/Desktop/Python/SalesForecast/models/"

# Tarih sütununu ayrıştırma
df['Tarih'] = pd.to_datetime(df['Tarih'])
df['Year'] = df['Tarih'].dt.year
df['Month'] = df['Tarih'].dt.month
df['Day'] = df['Tarih'].dt.day
df['dayofweek'] = df['Tarih'].dt.dayofweek
df['quarter'] = df['Tarih'].dt.quarter
df['weekofyear'] = df['Tarih'].dt.isocalendar().week
df['birim_fiyat'] = df['Birim Fiyat']

filterYear = True

if filterYear == True:
    df = df[df['Tarih'].dt.year == 2022]
else:
    pass

label_encoder = LabelEncoder()

imputer = SimpleImputer()   

df['Product_Code'] = label_encoder.fit_transform(df['Ürün Kodu'])

df_org = df.copy()

df = df.drop(['Yıl', 'Müşteri Kodu', 'Toplam', 'Net Tutar' , 'Tarih', 'Birim Fiyat'] , axis=1)

#%%

print(df.head(10))

df_train = df.copy()


    
#%%

from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor

print(df_train.head())

df_train['weekofyear'] = df_train['weekofyear'].astype(float)

print(df_train.dtypes)


X = df_train.drop(['Miktar','Ürün Kodu'], axis=1)
y = df_train["Miktar"]
# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


from sklearn.tree import DecisionTreeRegressor
model_file = "C:/Users/aysem/Desktop/Python/SalesForecast/models/"

def rComparison(names, results, cText=""):
    fig, ax = plt.subplots()
    ax.bar(names, results)
    ax.set_ylim([0.5, 1.0])
    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(results):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    #ape[~np.isfinite(ape)] = 0. # VERY questionable
    ape[~np.isfinite(ape)] = 1. # pessimist estimate
    return np.mean(ape)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

r2_list = {}
mae_list = {}
mse_list = {}
rmse_list = {}
mape_list = {}
wmape_list = {}

model_list = []

hyper_parameter = True
hyper_parameter = False

models = [ 
    (GradientBoostingRegressor(n_estimators=100, random_state=0), "GradientBoost", 'gb_model',
        {'n_estimators': [50, 100, 200],
         'learning_rate': [0.01, 0.1, 0.2],
         'max_depth': [3, 5, 7],
         'min_samples_split': [2, 5, 10]}),

    (LinearRegression(), "LinearReg", 'linear_reg_model',
        {'normalize': [True, False]}),

    (RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6), "RandomForest", 'rf_model',
        {'n_estimators': [50, 100, 200],
         'max_depth': [None, 10, 20, 30],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4]}),

    (DecisionTreeRegressor(random_state=0), "DecisionTree", 'dt_model',
        {'max_depth': [None, 10, 20, 30],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4]}),

    (RidgeCV(), "RidgeCV", 'ridgecv_model', {'alphas': [(0.1, 1.0, 10.0)]}),

    (KNeighborsRegressor(), "KNeighborsRegressor", 'kneighbors_model',
        {'n_neighbors': [3, 5, 7, 10],
         'weights': ['uniform', 'distance'],
         'metric': ['euclidean', 'manhattan']}),

    # (SVR(), "SVR", 'svr_model', {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),

    (ElasticNet(), "ElasticNet", 'elasticnet_model',
        {'alpha': [0.1, 0.5, 1.0],
         'l1_ratio': [0.2, 0.5, 0.8]}),

    (Lasso(), "Lasso", 'lasso_model',
        {'alpha': [0.1, 0.5, 1.0]}),

    (BaggingRegressor(), "BaggingRegressor", 'bagging_model',
        {'n_estimators': [10, 50, 100],
         'max_samples': [0.5, 1.0],
         'max_features': [0.5, 1.0]}),

    (MLPRegressor(), "MLPRegressor", 'mlpregressor_model',
        {'hidden_layer_sizes': [(50, 50), (100, 50, 100)],
         'activation': ['relu', 'tanh'],
         'alpha': [0.0001, 0.001, 0.01],
         'learning_rate': ['constant', 'invscaling', 'adaptive']}),
    
    # (xgb.XGBRegressor(n_estimators=100, random_state=0, n_jobs=6), "XGBoost", 'xgb_model',
    #     {'n_estimators': [50, 100, 200],
    #      'learning_rate': [0.01, 0.1, 0.2],
    #      'max_depth': [3, 5, 7],
    #      'min_child_weight': [1, 3, 5]}),
]


for model,name,model_type,grid_params in models:
    
    # save_model(model, str(name + RunID))
    
    delfault_params = model.get_params()
    print("Default parameters: "+str(delfault_params))
    
    if name == "LinearReg" or hyper_parameter == False:
        model.fit(X_train_imputed, y_train)
    else:
        grid_search = GridSearchCV(model, grid_params, scoring='neg_mean_absolute_error', cv=3)
        grid_search.fit(X_train_imputed, y_train)
        best_params = grid_search.best_params_
        model.fit(X_train_imputed, y_train)
        
        model = model.set_params(**best_params)
    
    print(name + "...eğitiliyor")
    model.fit(X_train_imputed, y_train)   
    print(name + "...eğitildi")
    
    save_name = model_file + str(RunID)+str(name)+".pkl"
    
    with open(save_name, 'wb') as file:
        pickle.dump((model, label_encoder), file)
    
    print(save_name + "...kaydedildi")
    
    y_pred = model.predict(X_test_imputed)
    
    
    # R^2 (Coefficient of Determination)
    r2 = r2_score(y_test, y_pred)
    print(f'R^2: {r2:.4f}')
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MAE: {mae:.4f}')
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse:.4f}')
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse:.4f}')

    r2_list["r2-"+str(name)] = r2
    mae_list["mae-"+str(name)] = mae
    mse_list["mse-"+str(name)] = mse
    rmse_list["rmse-"+str(name)] = rmse
        
    mape_val = mape(y_test, y_pred)
    wmape_val = wmape(y_test, y_pred)
    mape_list["mape-"+name] = mape_val
    wmape_list["wmape-"+name] = wmape_val
    
    print(f"{model_type.upper()} MAPE: {mape_val}")
    print(f"{model_type.upper()} WMAPE: {wmape_val}")
    
    
    model_list.append((model, name))

    
    import seaborn as sns
#%%

    if hasattr(model, 'feature_importances_'):
        iMesaj = "Feature Importances:\n\n"
        iMesaj += str(name) + " \n\n"
        importances = model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1][:10]
        top_importances = importances[indices]
        top_features = features[indices]
        plt.figure(figsize=(10,6))
        plt.title(f"{name} Feature Importances\n\n")
        sns.barplot(x=top_importances, y=top_features, orient='h')
        img_imp = name + RunID +"imp"+".png"
        plt.savefig(img_imp, format="png", bbox_inches="tight")
        plt.show()
        plt.close("all")
        for i in range(len(top_features)):
            iMesaj += f"{top_features[i]}: {top_importances[i]:.3f}" +"\n"
        

#%%
print(r2_list)
print(mae_list)
print(mse_list)
print(rmse_list)
print(mape_list)
print(wmape_list)


# rComparison(names, accuracy_results, "The accuracy score is a common metric used to evaluate the performance of a model in classification tasks. It measures the percentage of correct predictions made by the model across all classes.")

# Örnek bir tahmin yap
target_date = '2024-08-15'
product_code = 'Y30213'
birim_fiyat = 10

features = pd.DataFrame({
    'Year': [pd.to_datetime(target_date).year],
    'Month': [pd.to_datetime(target_date).month],
    'Day': [pd.to_datetime(target_date).day],
    'dayofweek': [pd.to_datetime(target_date).dayofweek],
    'quarter': [pd.to_datetime(target_date).quarter],
    'weekofyear': [pd.to_datetime(target_date).isocalendar()[1]],
    'birim_fiyat': [birim_fiyat],
    'Product_Code': [label_encoder.transform([product_code])[0]],
})

print("FEATURES: ",features)


for model, modelname in model_list:
    prediction = model.predict(features)[0]
    print(f"{target_date} tarihinde {product_code} ürün kodu için tahmini satış {modelname} modelinde: \n{prediction:.0f}")


# logging.shutdown()
#%%
import sys
from PyQt5 import QtWidgets
from baselay import Ui_baselay_MainWindow
# from prophet_1 import ProphetPredictor
import pickle
from sklearn.preprocessing import LabelEncoder

import pandas as pd
label_encoder = LabelEncoder()
import warnings



# Uyarıları kapat
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class MyApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_baselay_MainWindow()
        self.ui.setupUi(self)
        self.ui.hesapla_button.clicked.connect(self.hesapla_clicked)
        self.ui.prophet_button.clicked.connect(self.prophet_clicked)
        self.ui.xgb_button.clicked.connect(self.xgb_clicked)
        self.ui.random_forest_button.clicked.connect(self.random_forest_clicked)
        self.ui.gradient_boosting_button.clicked.connect(self.gradient_boosting_clicked)
        self.ui.mlp_button.clicked.connect(self.mlp_clicked)
        self.ui.lgbm_button.clicked.connect(self.lgbm_clicked)
        self.ui.ridgecv_button.clicked.connect(self.ridgecv_clicked)
        self.ui.lasso_button.clicked.connect(self.lasso_clicked)
        self.ui.decisiontree_button.clicked.connect(self.decisiontree_clicked)
        self.ui.knn_button.clicked.connect(self.knn_clicked)
        self.ui.bagging_button.clicked.connect(self.bagging_clicked)
        self.ui.linearreg_button.clicked.connect(self.linearreg_clicked)
        self.ui.elasticnet_button.clicked.connect(self.elasticnet_clicked)
        label_encoder.fit(df_org['Ürün Kodu'])

    def hesapla_clicked(self):
        urun_kodu = self.ui.urun_kodu.text()
        label_urun_kodu = label_encoder.transform([urun_kodu])[0]
        son_tarih = self.ui.son_tarih.date().toString("yyyy-MM-dd")
        birim_fiyat_button = float(self.ui.birim_fiyat_button.text())
        print(urun_kodu)
        print(son_tarih)
        print(birim_fiyat_button)
        label_urun_kodu = label_encoder.transform([urun_kodu])[0]

        

        # predictor = ProphetPredictor()
        # predicted_value = predictor.predict(urun_kodu, son_tarih)
        # self.ui.tablo.setItem(0, 0, QtWidgets.QTableWidgetItem(str(predicted_value)))
        # try:
        #     label_encoder.fit([urun_kodu])
        # except Exception as e:
        #     print(f'Hata: {e}')
            
        orijinal_urun_kodu = label_encoder.inverse_transform([label_urun_kodu])[0]
        print("Label ürün kodu: ", label_urun_kodu)
        print("Gerçek ürün kodu: ", orijinal_urun_kodu)

        features = pd.DataFrame({
            'Year': [pd.to_datetime(son_tarih).year],
            'Month': [pd.to_datetime(son_tarih).month],
            'Day': [pd.to_datetime(son_tarih).day],
            'dayofweek': [pd.to_datetime(son_tarih).dayofweek],
            'quarter': [pd.to_datetime(son_tarih).quarter],
            'weekofyear': [pd.to_datetime(son_tarih).isocalendar()[1]],
            'birim_fiyat': [birim_fiyat_button],
            'Product_Code': [label_urun_kodu],  # Burada etiketlenmiş ürün kodunu kullanın
        })
        i = 1
        for model, modelname in model_list:
            prediction = model.predict(features)[0]
            print(f"{son_tarih} tarihinde {urun_kodu} ürün kodu için tahmini satış {modelname} modelinde: \n{prediction:.0f}")
        
            prediction = model.predict(features)[0]
            self.ui.tablo.setItem(0, i + 1, QtWidgets.QTableWidgetItem(str(int(prediction))))
            i = i + 1

        
    def prophet_clicked(self):
        print("prophet")
        
    def xgb_clicked(self):
        print("xgb")
    
        
    def random_forest_clicked(self):
        print("random_forest")
        
        
    def gradient_boosting_clicked(self):
        print("gradient_boosting")    
        
    def mlp_clicked(self):
        print("mlp")    
        
        
    def lgbm_clicked(self):
        print("lgbm_clicked")        
            
    def ridgecv_clicked(self):
        print("ridgecv_clicked")   
        
    def lasso_clicked(self):
        print("lasso_clicked")     

    def decisiontree_clicked(self):
        print("decisiontree_clicked")        

    def bagging_clicked(self):
        print("bagging_clicked")     
     
    def linearreg_clicked(self):
        print("linearreg_clicked")     
        
    def elasticnet_clicked(self):
        print("elasticnet_clicked")         
        
        
    def knn_clicked(self):
        print("knn_clicked")        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApplication()
    window.show()
    sys.exit(app.exec_())
    
    
#%% Hazır Model Yükle
import sys
from PyQt5 import QtWidgets
from baselay import Ui_baselay_MainWindow
# from prophet_1 import ProphetPredictor
import pickle
from sklearn.preprocessing import LabelEncoder

import pandas as pd
label_encoder = LabelEncoder()
import warnings
from datetime import datetime


RunID = datetime.now().strftime("%Y%m%d-%H%M")

pd.set_option("display.max_columns", None)

# Veriyi hazırla
file_path = "new_filtered.xlsx"
df = pd.read_excel(file_path)

filterYear = True

model_file = "C:/Users/aysem/Desktop/Python/SalesForecast/models/"

# Tarih sütununu ayrıştırma
df['Tarih'] = pd.to_datetime(df['Tarih'])
df['Year'] = df['Tarih'].dt.year
df['Month'] = df['Tarih'].dt.month
df['Day'] = df['Tarih'].dt.day
df['dayofweek'] = df['Tarih'].dt.dayofweek
df['quarter'] = df['Tarih'].dt.quarter
df['weekofyear'] = df['Tarih'].dt.isocalendar().week
df['birim_fiyat'] = df['Birim Fiyat']

if filterYear == True:
    df = df[df['Tarih'].dt.year == 2022]
else:
    pass

label_encoder = LabelEncoder()

imputer = SimpleImputer()   

df['Product_Code'] = label_encoder.fit_transform(df['Ürün Kodu'])

df_org = df.copy()

df = df.drop(['Yıl', 'Müşteri Kodu', 'Toplam', 'Net Tutar' , 'Tarih', 'Birim Fiyat'] , axis=1)

models = [
    "20231227-2000XGBoost.pkl",
    "20231227-2000GradientBoost.pkl",
    "20231227-2000MLPRegressor.pkl",
    "20231227-2000LightGBM.pkl",
    "20231227-2000RidgeCV.pkl",
    "20231227-2000Lasso.pkl",
    "20231227-2000DecisionTree.pkl",
    "20231227-2000KNeighborsRegressor.pkl",
    "20231227-2000LinearReg.pkl",      
    "20231227-2000ElasticNet.pkl",
    # "20231227-2000BaggingRegressor.pkl", 
    # "20231228-2000RandomForest.pkl", 
    ]

# Uyarıları kapat
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



class MyApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
       
        self.label_encoder = LabelEncoder()
        self.models = []

        self.label_encoder.fit(df_org['Ürün Kodu'])

        for model_name in models:
            model_path = f"models/{model_name}"
            with open(model_path, 'rb') as file:
                loaded_model, _ = pickle.load(file)
                self.models.append(loaded_model)
                
        self.ui = Ui_baselay_MainWindow()
        self.ui.setupUi(self)
        self.ui.hesapla_button.clicked.connect(self.hesapla_clicked)
        self.ui.prophet_button.clicked.connect(self.prophet_clicked)
        self.ui.xgb_button.clicked.connect(self.xgb_clicked)
        self.ui.random_forest_button.clicked.connect(self.random_forest_clicked)
        self.ui.gradient_boosting_button.clicked.connect(self.gradient_boosting_clicked)
        self.ui.mlp_button.clicked.connect(self.mlp_clicked)
        self.ui.lgbm_button.clicked.connect(self.lgbm_clicked)
        self.ui.ridgecv_button.clicked.connect(self.ridgecv_clicked)
        self.ui.lasso_button.clicked.connect(self.lasso_clicked)
        self.ui.decisiontree_button.clicked.connect(self.decisiontree_clicked)
        self.ui.knn_button.clicked.connect(self.knn_clicked)
        self.ui.bagging_button.clicked.connect(self.bagging_clicked)
        self.ui.linearreg_button.clicked.connect(self.linearreg_clicked)
        self.ui.elasticnet_button.clicked.connect(self.elasticnet_clicked)
        label_encoder.fit(df_org['Ürün Kodu'])

    def hesapla_clicked(self):
        urun_kodu = self.ui.urun_kodu.text()
        son_tarih = self.ui.son_tarih.date().toString("yyyy-MM-dd")
        birim_fiyat_button = float(self.ui.birim_fiyat_button.text())
        print(urun_kodu)
        print(son_tarih)
        print(birim_fiyat_button)
        label_urun_kodu = self.label_encoder.transform([urun_kodu])[0]
        
        print("Label ürün kodu: ", label_urun_kodu)
        print("Gerçek ürün kodu: ", urun_kodu)

        

        # predictor = ProphetPredictor()
        # predicted_value = predictor.predict(urun_kodu, son_tarih)
        # self.ui.tablo.setItem(0, 0, QtWidgets.QTableWidgetItem(str(predicted_value)))
        # try:
        #     label_encoder.fit([urun_kodu])
        # except Exception as e:
        #     print(f'Hata: {e}')
            
        orijinal_urun_kodu = label_encoder.inverse_transform([label_urun_kodu])[0]
        print("Label ürün kodu: ", label_urun_kodu)
        print("Gerçek ürün kodu: ", orijinal_urun_kodu)

        features = pd.DataFrame({
            'Year': [pd.to_datetime(son_tarih).year],
            'Month': [pd.to_datetime(son_tarih).month],
            'Day': [pd.to_datetime(son_tarih).day],
            'dayofweek': [pd.to_datetime(son_tarih).dayofweek],
            'quarter': [pd.to_datetime(son_tarih).quarter],
            'weekofyear': [pd.to_datetime(son_tarih).isocalendar()[1]],
            'birim_fiyat': [birim_fiyat_button],
            'Product_Code': [label_urun_kodu],  # Burada etiketlenmiş ürün kodunu kullanın
        })
        
        i = 1
        for model in self.models:
            prediction = model.predict(features)[0]
            self.ui.tablo.setItem(0, i + 1, QtWidgets.QTableWidgetItem(str(int(prediction))))
            i = i + 1

        
    def prophet_clicked(self):
        print("prophet")
        
    def xgb_clicked(self):
        print("xgb")
    
        
    def random_forest_clicked(self):
        print("random_forest")
        
        
    def gradient_boosting_clicked(self):
        print("gradient_boosting")    
        
    def mlp_clicked(self):
        print("mlp")    
        
        
    def lgbm_clicked(self):
        print("lgbm_clicked")        
            
    def ridgecv_clicked(self):
        print("ridgecv_clicked")    
        
    def lasso_clicked(self):
        print("lasso_clicked")        

    def decisiontree_clicked(self):
        print("decisiontree_clicked")        

    def bagging_clicked(self):
        print("bagging_clicked")     
     
    def linearreg_clicked(self):
        print("linearreg_clicked")     
        
    def elasticnet_clicked(self):
        print("elasticnet_clicked")         
        
        
    def knn_clicked(self):
        print("knn_clicked")        

 

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApplication()
    window.show()
    sys.exit(app.exec_())
