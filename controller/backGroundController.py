import cv2
import numpy as np
import glob
from time import time
import matplotlib.pyplot as plt

class BackGroundController:

    def __init__(self,routeMainImage): 
        self.routeMainImage = routeMainImage
        self.time = []
        self.routeMain = self.getRouteChange(routeMainImage)
        self.initial()
        print(self.time)

    def initial(self):
        self.mainImage = cv2.imread(self.routeMainImage,0)
        self.fondo     = self.mainImage
        self.selectMask()

    def selectMask(self):
        self.cut = cv2.selectROI(self.mainImage)
        self.validated = self.validateMask()
        self.mask = self.createMask(self.cut[0],self.validated[0],self.cut[1],self.validated[1])
        cv2.destroyAllWindows()
    
    def validateMask(self):
        x = self.cut[2]
        y = self.cut[3]
        if(x%2==0):
            x+=1
        if(y%2==0):
            y+=1
        return x,y

    def createMask(self,xinitial,xfinish,yinitial,yfinish):
        return self.mainImage[int(yinitial):int(yinitial+yfinish),int(xinitial):int(xinitial+xfinish)]
    
    def ssd(self,alpha,numberImages):
        self.fondo = self.mainImage
        elapsed_time = 0
        start_time = time()
        for w in range(numberImages):
            imageN = cv2.imread(self.getNameImage(w),0)
            rest = 99**99
            pos = []
            for y in range(imageN.shape[0]-self.mask.shape[0]):
                for x in range(imageN.shape[1]-self.mask.shape[1]):
                    sum = 0
                    cut = self.cutImage(imageN,y,x) #correlación
                    sum = np.sum((cut-self.mask)**2) #ssd
                    if sum < rest :
                        rest = sum
                        pos = x,y
            desx =  self.cut[0] - pos[0]
            desy =  self.cut[1] - pos[1] 
            m = np.float32([[1,0,desx],[0,1,desy]])
            dst = cv2.warpAffine(imageN,m,(imageN.shape[1],imageN.shape[0]))
            elapsed_time += time() - start_time
            self.extraerFondo(dst,alpha,'ssd',w)

        cv2.imwrite("ssd/ssd-result.jpg",self.fondo)
        self.generateVideo("ssd")
        elapsed_time += time() - start_time
        self.time.append(elapsed_time)
    
    def sad(self,alpha,numberImages):
        self.fondo     = self.mainImage
        elapsed_time = 0
        start_time = time()
        for w in range(numberImages):
            imageN = cv2.imread(self.getNameImage(w),0)
            rest = 99**99
            pos = []
            for y in range(imageN.shape[0]-self.mask.shape[0]):
                for x in range(imageN.shape[1]-self.mask.shape[1]):
                    sum = 0
                    cut = self.cutImage(imageN,y,x)
                    sum = np.sum(abs((cut-self.mask))) #sad
                    if sum < rest :
                        rest = sum
                        pos = x,y
            desx =  self.cut[0] - pos[0]
            desy =  self.cut[1] - pos[1] 
            m = np.float32([[1,0,desx],[0,1,desy]])
            dst = cv2.warpAffine(imageN,m,(imageN.shape[1],imageN.shape[0]))
            elapsed_time += time() - start_time
            self.extraerFondo(dst,alpha,'sad',w)
            
        cv2.imwrite("sad/sad-result.jpg",self.fondo)
        self.generateVideo("sad")
       
        self.time.append(elapsed_time)
    
    def cross(self, alpha, numberImages):
        self.fondo     = self.mainImage
        elapsed_time = 0
        start_time = time()
        for w in range(numberImages):
            imageN = cv2.imread(self.getNameImage(w),0)
            rest = 99**99
            pos = []
            for y in range(imageN.shape[0]-self.mask.shape[0]):
                for x in range(imageN.shape[1]-self.mask.shape[1]):
                    sum = 0
                    cut = self.cutImage(imageN,y,x)
                    sum = np.sum(cut*self.mask) #cross
                    if sum < rest :
                        rest = sum
                        pos = x,y
            desx =  self.cut[0] - pos[0]
            desy =  self.cut[1] - pos[1] 
            m = np.float32([[1,0,desx],[0,1,desy]])
            dst = cv2.warpAffine(imageN,m,(imageN.shape[1],imageN.shape[0]))
            elapsed_time += time() - start_time
            self.extraerFondo(dst,alpha,'cross',w)
            
        cv2.imwrite("cross/cross-result.jpg",self.fondo)
        self.generateVideo("cross")
        
        self.time.append(elapsed_time)

        fig = plt.figure(u'Gráfica de tiempos') # Figure
        ax = fig.add_subplot(111) # Axes
        plt.ylabel("Tiempo de ejecuciion")
        plt.xlabel("Metodos de extracion de fondo")
        plt.title("Tiempo de ejecución de recuperación de fondo("+str(numberImages)+" Imagenes)")
        nombres = ['SSD','SAD','CROSS']
        datos = [self.time[0],self.time[1],self.time[2]]
        xx = range(len(datos))
        ax.bar(xx, datos, width=0.8, align='center')
        ax.set_xticks(xx)
        ax.set_xticklabels(nombres)

        plt.show()

    def extraerFondo(self,alinear,alpha,method,w):
        self.fondo = (1-alpha)*self.fondo+(alpha*alinear)
        nameImageResult = method+"/show_{}.jpg".format(w)
        cv2.imwrite(nameImageResult,self.fondo)
        print("Imagen guardada numero: ",w)
            
    def cutImage(self,imageN,y,x):
        return imageN[int(y):int(y+self.mask.shape[0]),int(x):int(x+self.mask.shape[1])]

    def getNameImage(self,index):
        if index < 8:
            namei = self.routeMain+"im00000"+repr(index+2)+".jpg"
        elif index < 98:
            namei = self.routeMain+"im0000"+repr(index+2)+".jpg"
        else:
            namei = self.routeMain+"im000"+repr(index+2)+".jpg"
        return namei

    def getRouteChange(self,route):
        newRoute = ""
        data  = route.split("/")
        size  = len(data)-1
        for i in range(size):
            newRoute += data[i]+"/"
        return newRoute

    def generateVideo(self,method):
        img_array = []
        for filename in glob.glob('C:/Users/usuario/Desktop/IA-C1E1/'+method+'/*.jpg'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter(method+'/'+method+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()  