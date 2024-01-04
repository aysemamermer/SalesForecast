# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'baselay.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_baselay_MainWindow(object):
    def setupUi(self, baselay_MainWindow):
        baselay_MainWindow.setObjectName("baselay_MainWindow")
        baselay_MainWindow.resize(1000, 753)
        baselay_MainWindow.setMinimumSize(QtCore.QSize(1000, 753))
        baselay_MainWindow.setMaximumSize(QtCore.QSize(1000, 753))
        baselay_MainWindow.setStyleSheet("background-color: rgb(15, 33, 103);")
        self.centralwidget = QtWidgets.QWidget(baselay_MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(708, 570))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.urun_kodu = QtWidgets.QLineEdit(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.urun_kodu.sizePolicy().hasHeightForWidth())
        self.urun_kodu.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.urun_kodu.setFont(font)
        self.urun_kodu.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.urun_kodu.setText("")
        self.urun_kodu.setObjectName("urun_kodu")
        self.horizontalLayout.addWidget(self.urun_kodu)
        self.birim_fiyat_button = QtWidgets.QLineEdit(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.birim_fiyat_button.sizePolicy().hasHeightForWidth())
        self.birim_fiyat_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.birim_fiyat_button.setFont(font)
        self.birim_fiyat_button.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.birim_fiyat_button.setObjectName("birim_fiyat_button")
        self.horizontalLayout.addWidget(self.birim_fiyat_button)
        self.son_tarih = QtWidgets.QDateEdit(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(self.son_tarih.sizePolicy().hasHeightForWidth())
        self.son_tarih.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.son_tarih.setFont(font)
        self.son_tarih.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.son_tarih.setDateTime(QtCore.QDateTime(QtCore.QDate(2017, 2, 1), QtCore.QTime(0, 0, 0)))
        self.son_tarih.setMinimumDate(QtCore.QDate(2017, 2, 1))
        self.son_tarih.setObjectName("son_tarih")
        self.horizontalLayout.addWidget(self.son_tarih)
        self.hesapla_button = QtWidgets.QPushButton(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hesapla_button.sizePolicy().hasHeightForWidth())
        self.hesapla_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.hesapla_button.setFont(font)
        self.hesapla_button.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.hesapla_button.setIconSize(QtCore.QSize(50, 25))
        self.hesapla_button.setObjectName("hesapla_button")
        self.horizontalLayout.addWidget(self.hesapla_button)
        spacerItem = QtWidgets.QSpacerItem(390, 55, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.gridLayout.addWidget(self.widget_3, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 5, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 5, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addItem(spacerItem3, 2, 1, 1, 1)
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.grafik = QtWidgets.QGraphicsView(self.widget_4)
        self.grafik.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.grafik.setObjectName("grafik")
        self.verticalLayout.addWidget(self.grafik)
        self.gridLayout.addWidget(self.widget_4, 4, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tablo = QtWidgets.QTableWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.tablo.sizePolicy().hasHeightForWidth())
        self.tablo.setSizePolicy(sizePolicy)
        self.tablo.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.tablo.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tablo.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tablo.setObjectName("tablo")
        self.tablo.setColumnCount(13)
        self.tablo.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablo.setHorizontalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        item.setFont(font)
        self.tablo.setItem(0, 2, item)
        self.tablo.horizontalHeader().setCascadingSectionResizes(False)
        self.tablo.horizontalHeader().setSortIndicatorShown(False)
        self.tablo.horizontalHeader().setStretchLastSection(True)
        self.tablo.verticalHeader().setCascadingSectionResizes(False)
        self.tablo.verticalHeader().setHighlightSections(False)
        self.tablo.verticalHeader().setSortIndicatorShown(False)
        self.tablo.verticalHeader().setStretchLastSection(True)
        self.horizontalLayout_2.addWidget(self.tablo)
        self.gridLayout.addWidget(self.widget, 3, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 4, 3, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        self.gridLayout.addItem(spacerItem5, 6, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem6, 4, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(10, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addItem(spacerItem7, 0, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 1, 0, 1, 1)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mlp_button = QtWidgets.QPushButton(self.widget_2)
        self.mlp_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.mlp_button.setObjectName("mlp_button")
        self.gridLayout_2.addWidget(self.mlp_button, 1, 2, 1, 1)
        self.decisiontree_button = QtWidgets.QPushButton(self.widget_2)
        self.decisiontree_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.decisiontree_button.setObjectName("decisiontree_button")
        self.gridLayout_2.addWidget(self.decisiontree_button, 2, 3, 1, 1)
        self.gradient_boosting_button = QtWidgets.QPushButton(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.gradient_boosting_button.sizePolicy().hasHeightForWidth())
        self.gradient_boosting_button.setSizePolicy(sizePolicy)
        self.gradient_boosting_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.gradient_boosting_button.setObjectName("gradient_boosting_button")
        self.gridLayout_2.addWidget(self.gradient_boosting_button, 0, 3, 1, 1)
        self.linearreg_button = QtWidgets.QPushButton(self.widget_2)
        self.linearreg_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.linearreg_button.setObjectName("linearreg_button")
        self.gridLayout_2.addWidget(self.linearreg_button, 3, 3, 1, 1)
        self.ridgecv_button = QtWidgets.QPushButton(self.widget_2)
        self.ridgecv_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.ridgecv_button.setObjectName("ridgecv_button")
        self.gridLayout_2.addWidget(self.ridgecv_button, 2, 1, 1, 1)
        self.lasso_button = QtWidgets.QPushButton(self.widget_2)
        self.lasso_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.lasso_button.setObjectName("lasso_button")
        self.gridLayout_2.addWidget(self.lasso_button, 2, 2, 1, 1)
        self.bagging_button = QtWidgets.QPushButton(self.widget_2)
        self.bagging_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.bagging_button.setObjectName("bagging_button")
        self.gridLayout_2.addWidget(self.bagging_button, 3, 2, 1, 1)
        self.prophet_button = QtWidgets.QPushButton(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.prophet_button.sizePolicy().hasHeightForWidth())
        self.prophet_button.setSizePolicy(sizePolicy)
        self.prophet_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.prophet_button.setObjectName("prophet_button")
        self.gridLayout_2.addWidget(self.prophet_button, 0, 1, 1, 1)
        self.xgb_button = QtWidgets.QPushButton(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.xgb_button.sizePolicy().hasHeightForWidth())
        self.xgb_button.setSizePolicy(sizePolicy)
        self.xgb_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.xgb_button.setObjectName("xgb_button")
        self.gridLayout_2.addWidget(self.xgb_button, 0, 2, 1, 1)
        self.knn_button = QtWidgets.QPushButton(self.widget_2)
        self.knn_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.knn_button.setObjectName("knn_button")
        self.gridLayout_2.addWidget(self.knn_button, 3, 1, 1, 1)
        self.lgbm_button = QtWidgets.QPushButton(self.widget_2)
        self.lgbm_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.lgbm_button.setObjectName("lgbm_button")
        self.gridLayout_2.addWidget(self.lgbm_button, 1, 3, 1, 1)
        self.random_forest_button = QtWidgets.QPushButton(self.widget_2)
        self.random_forest_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.random_forest_button.setObjectName("random_forest_button")
        self.gridLayout_2.addWidget(self.random_forest_button, 1, 1, 1, 1)
        self.elasticnet_button = QtWidgets.QPushButton(self.widget_2)
        self.elasticnet_button.setStyleSheet("background-color: rgb(255, 236, 214);")
        self.elasticnet_button.setObjectName("elasticnet_button")
        self.gridLayout_2.addWidget(self.elasticnet_button, 4, 2, 1, 1)
        self.gridLayout.addWidget(self.widget_2, 5, 1, 1, 1)
        baselay_MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(baselay_MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        baselay_MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(baselay_MainWindow)
        self.statusbar.setObjectName("statusbar")
        baselay_MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(baselay_MainWindow)
        QtCore.QMetaObject.connectSlotsByName(baselay_MainWindow)

    def retranslateUi(self, baselay_MainWindow):
        _translate = QtCore.QCoreApplication.translate
        baselay_MainWindow.setWindowTitle(_translate("baselay_MainWindow", "BASELAY Forecasting"))
        self.urun_kodu.setPlaceholderText(_translate("baselay_MainWindow", "Ürün Kodu Gİriniz..."))
        self.birim_fiyat_button.setPlaceholderText(_translate("baselay_MainWindow", "Birim Fiyat (Opsiyonel)"))
        self.hesapla_button.setText(_translate("baselay_MainWindow", "HESAPLA"))
        item = self.tablo.verticalHeaderItem(0)
        item.setText(_translate("baselay_MainWindow", "Miktar "))
        item = self.tablo.horizontalHeaderItem(0)
        item.setText(_translate("baselay_MainWindow", "Prophet"))
        item = self.tablo.horizontalHeaderItem(1)
        item.setText(_translate("baselay_MainWindow", "XGB"))
        item = self.tablo.horizontalHeaderItem(2)
        item.setText(_translate("baselay_MainWindow", "Gradient Boost"))
        item = self.tablo.horizontalHeaderItem(3)
        item.setText(_translate("baselay_MainWindow", "Random Forest"))
        item = self.tablo.horizontalHeaderItem(4)
        item.setText(_translate("baselay_MainWindow", "MLP"))
        item = self.tablo.horizontalHeaderItem(5)
        item.setText(_translate("baselay_MainWindow", "LGBM"))
        item = self.tablo.horizontalHeaderItem(6)
        item.setText(_translate("baselay_MainWindow", "RidgeCV"))
        item = self.tablo.horizontalHeaderItem(7)
        item.setText(_translate("baselay_MainWindow", "Lasso"))
        item = self.tablo.horizontalHeaderItem(8)
        item.setText(_translate("baselay_MainWindow", "DecisionTree"))
        item = self.tablo.horizontalHeaderItem(9)
        item.setText(_translate("baselay_MainWindow", "KNN"))
        item = self.tablo.horizontalHeaderItem(10)
        item.setText(_translate("baselay_MainWindow", "Bagging"))
        item = self.tablo.horizontalHeaderItem(11)
        item.setText(_translate("baselay_MainWindow", "LinearReg"))
        item = self.tablo.horizontalHeaderItem(12)
        item.setText(_translate("baselay_MainWindow", "ElasticNet"))
        __sortingEnabled = self.tablo.isSortingEnabled()
        self.tablo.setSortingEnabled(False)
        self.tablo.setSortingEnabled(__sortingEnabled)
        self.mlp_button.setText(_translate("baselay_MainWindow", "MLP"))
        self.decisiontree_button.setText(_translate("baselay_MainWindow", "Decision Tree"))
        self.gradient_boosting_button.setText(_translate("baselay_MainWindow", "Gradient Boosting"))
        self.linearreg_button.setText(_translate("baselay_MainWindow", "Linear Reg"))
        self.ridgecv_button.setText(_translate("baselay_MainWindow", "RidgeCV"))
        self.lasso_button.setText(_translate("baselay_MainWindow", "Lasso"))
        self.bagging_button.setText(_translate("baselay_MainWindow", "Bagging"))
        self.prophet_button.setText(_translate("baselay_MainWindow", "Prophet"))
        self.xgb_button.setText(_translate("baselay_MainWindow", "XGB"))
        self.knn_button.setText(_translate("baselay_MainWindow", "KNN"))
        self.lgbm_button.setText(_translate("baselay_MainWindow", "LGBM"))
        self.random_forest_button.setText(_translate("baselay_MainWindow", "Random Forest"))
        self.elasticnet_button.setText(_translate("baselay_MainWindow", "Elastic Net"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    baselay_MainWindow = QtWidgets.QMainWindow()
    ui = Ui_baselay_MainWindow()
    ui.setupUi(baselay_MainWindow)
    baselay_MainWindow.show()
    sys.exit(app.exec_())
