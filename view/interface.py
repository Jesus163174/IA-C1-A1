from PyQt5 import QtCore, QtGui, QtWidgets

class Interface(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1258, 472)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMaximumSize(QtCore.QSize(1258, 472))

        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.scroll_image_before = QtWidgets.QScrollArea(Dialog)
        self.scroll_image_before.setGeometry(QtCore.QRect(40, 50, 521, 401))
        self.scroll_image_before.setWidgetResizable(True)
        self.scroll_image_before.setObjectName("scroll_image_before")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 519, 399))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scroll_image_before.setWidget(self.scrollAreaWidgetContents)
    
        self.scroll_image_after = QtWidgets.QScrollArea(Dialog)
        self.scroll_image_after.setGeometry(QtCore.QRect(680, 50, 521, 401))
        self.scroll_image_after.setWidgetResizable(True)
        self.scroll_image_after.setObjectName("scroll_image_after")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 519, 399))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.scroll_image_after.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(560, 160, 91, 161))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        font = QtGui.QFont()
        font.setPointSize(12)
        
        font = QtGui.QFont()
        font.setPointSize(12)
        self.buttonSelect = QtWidgets.QPushButton(Dialog)
        self.buttonSelect.setGeometry(QtCore.QRect(130, 20, 89, 23))
        self.buttonSelect.setObjectName("buttonSelect")

        self.SSD = QtWidgets.QPushButton(Dialog)
        self.SSD.setGeometry(QtCore.QRect(573, 265, 89, 23))
        self.SSD.setObjectName("SSD")

        self.SAD = QtWidgets.QPushButton(Dialog)
        self.SAD.setGeometry(QtCore.QRect(573, 300, 89, 23))
        self.SAD.setObjectName("SAD")

        self.CRR = QtWidgets.QPushButton(Dialog)
        self.CRR.setGeometry(QtCore.QRect(573, 335, 89, 23))
        self.CRR.setObjectName("CRR")

        self.textNImagenes = QtWidgets.QLineEdit(Dialog)
        
        self.textNImagenes.setObjectName("textNImagenes")
        self.textNImagenes.setGeometry(QtCore.QRect(630, 212, 45, 30))
        
        self.textGama = QtWidgets.QLineEdit(Dialog)
        self.textGama.setObjectName("textGama")
        self.textGama.setGeometry(QtCore.QRect(620, 165, 50, 20))

        
        self.gama = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.gama.setFont(font)
        self.gama.setObjectName("gama")
        self.numeroImagene = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.numeroImagene.setFont(font)
        self.numeroImagene.setObjectName("numeroImagene")
        # self.gridLayout.addWidget(self.numeroImagene, 0, 0, 0, 0)
        self.numeroImagene.setGeometry(QtCore.QRect(562, 180, 65, 90))
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Extracion de Fondo"))
        self.buttonSelect.setText(_translate("Dialog", "Importar Imagen"))
        self.SSD.setText(_translate("Dialog", "Metodo SSD"))
        self.SAD.setText(_translate("Dialog", "Metodo SAD"))
        self.CRR.setText(_translate("Dialog", "Metodo CRR"))
        self.gama.setText(_translate("Dialog", "Alpha: "))
        self.numeroImagene.setText(_translate("Dialog", "Numero de\nImagenes: "))