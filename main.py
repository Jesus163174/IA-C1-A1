from view.interface import *
from view.interface import Interface
from controller.mainController import MainController

app    = QtWidgets.QApplication([])
window =  MainController()
window.show()
app.exec_()



