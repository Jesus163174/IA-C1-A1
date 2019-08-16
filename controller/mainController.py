from controller.backGroundController import BackGroundController
from view.interface import *
from view.interface import Interface

class MainController(QtWidgets.QMainWindow,Interface):

    def __init__(self, *args, **kwargs):
        self.routeImage = ""
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.buttonSelect.clicked.connect(self.selectImage)
        self.SSD.clicked.connect(self.ssd)
        self.SAD.clicked.connect(self.sad)
        self.CRR.clicked.connect(self.cross)

    #metodo para seleccionar la imagen principal
    def selectImage(self):
        imageLabel = QtWidgets.QLabel()
        self.routeImage , _ = QtWidgets.QFileDialog.getOpenFileName(None,
        'Select Image', '', "Image files (*.jpg *.png *.jpeg)")
        if self.routeImage:
            pixmap = QtGui.QPixmap(self.routeImage)    
            imageLabel.setPixmap(pixmap)
            self.scroll_image_before.setBackgroundRole(QtGui.QPalette.Dark)
            self.scroll_image_before.setWidget(imageLabel)
        self.background = BackGroundController(self.routeImage) 
    
    def ssd(self):
        numberImages = int(self.textNImagenes.text())
        alpha = float(self.textGama.text())
        self.background.ssd(alpha,numberImages)
    
    def sad(self):
        numberImages = int(self.textNImagenes.text())
        alpha = float(self.textGama.text())
        self.background.sad(alpha,numberImages)
    def cross(self):
        numberImages = int(self.textNImagenes.text())
        alpha = float(self.textGama.text())
        self.background.cross(alpha,numberImages)
    
        