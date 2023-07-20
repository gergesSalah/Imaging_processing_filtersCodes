import tkinter
from tkinter import*
from PIL import Image, ImageTk
from PIL import Image,ImageFilter
import tkinter as TK
from tkinter import messagebox
import numpy as np
import cv2
#from playsound import playsound
import os
import time
import HandTracking as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#song
# Everything stops when played
os.system("Sam21.wav")

# Everything continues when played
os.system("Sam21.wav&")

#frame
root=Tk()
root.title('Imaging processing')
root.geometry("1218x700")
C = Canvas(root,bg='#595959')      
C.pack(fill=BOTH,expand=1)

#img
image1 = Image.open("am5.jpeg")
test = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(image=test)
label1.image = test

# Position image
label1.place(x=-10, y=0)


#design
image1 = Image.open("W3.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=0, y=558)

image1 = Image.open("W3.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=167, y=558)

image1 = Image.open("W3.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=524, y=606)

image1 = Image.open("W3.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=533, y=606)

image1 = Image.open("W3.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=766, y=654)


#the image ofconstract function
#in functioin scope
#img for max and min fliter
imgO = Image.open("am5.jpeg")
#global variable for functions
img = cv2.imread('am5.jpeg')
#for intensity function
imgI = cv2.imread('am5.jpeg',0)

#functions
def guassian():
    dts = cv2.GaussianBlur(img,(35,35),40)
    #cv2.imshow("before Gaussian",img)
    cv2.imshow("After Gaussian",dts)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def Negative():
    row,cols,chan = img.shape
    img2 = np.zeros((row,cols,chan),np.uint8)
    for i in range(row):
        for j in range(cols):
            img2[i,j] = 255 - img[i,j]        
    cv2.imshow("after negative",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def logTransmition():    
    row,col,chan = img.shape
    img2 = np.zeros(img.shape,np.uint8)
    for i in range(row):
        for j in range(col):
            img2[i,j] = 85 * np.log(img[i,j])
    cv2.imshow("after logTrans",img2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def powerLow():
    row,cols,chan = img.shape
    img2 = np.zeros((row,cols,chan),np.uint8)
    for i in range(row):
        for j in range(cols):
            img2[i,j] = 17 * pow(img[i,j],0.5)    
    cv2.imshow("after powerLow",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Blur():
    dts = cv2.blur(img,(9,9))
    cv2.imshow("After",dts)
    cv2.waitKey()
    cv2.destroyAllWindows()

def copy():
    row,col,chan = img.shape
    img2 = np.zeros(img.shape,np.uint8)
    for i in range(row):
        for j in range(col):
            img2[i,j] = img[i,j]
    cv2.imwrite("copied image.jpg",img2)
    messagebox.showinfo('info',"the copied photo in programme file")
    cv2.imshow("after copy",img2)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
def MAX():
    dts = imgO.filter(ImageFilter.MaxFilter(size=3))
    dts.show("after")
    cv2.waitKey()
    cv2.destroyAllWindows()

def MIN():
    dts = imgO.filter(ImageFilter.MinFilter(size=3))
    dts.show("after")
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def SMOTH():
    karnel = np.ones((9,9),np.float32)/25 #why we devision on 25 , and what is float32
    dts = cv2.filter2D(img,-1,karnel)
    cv2.imshow("After smoth",dts)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
#def Rectangle():
 #   startpoint = (30,30)
  #  endpoint = (60,60)
   # col = (255,0,0)
  #  thick = 2
#    newimage = cv2.rectangle(img,startpoint,endpoint,col,thick)
 #   cv2.imshow("rectangle",newimage)
  #  cv2.waitKey(0)
   # cv2.destroyAllWindows()

#def Text():
 #   org = (50,50)
  #  font = cv2.FONT_HERSHEY_COMPLEX
   # fontScale = 2
    #color = (40,255,0)
  #  thick = 2
#    newImage = cv2.putText(img,"-----",org,font,fontScale,color,thick)
 #   cv2.imshow("after add Text",newImage)
  #  cv2.waitKey(0)
   # cv2.destroyAllWindows()

def sharp():
    dst = cv2.blur(img,(9,9))

    sub = cv2.subtract(img,dst)

    row,col,chan = sub.shape
    #maltiplay this sup by num
    subFac = np.zeros(sub.shape,np.uint8)
    for i in range(row):
        for j in range(col):
            subFac[i,j] = sub[i,j]*3 
            
    fin = cv2.addWeighted(subFac,0.5,img,0.9,0)
    
    messagebox.showinfo('information',"this function is not required but we made it by self")
    cv2.imshow("after enhancment",fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Histogram():
    src = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    histimage = cv2.equalizeHist(src)
    cv2.imshow("After histogram",histimage)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def intensity():
    row, column = imgI.shape
    img1 = np.zeros((row,column),dtype = 'uint8')
    min_range = 10
    max_range = 60
    for i in range(row):
        for j in range(column):
            if imgI[i,j]>min_range and imgI[i,j]<max_range:
                img1[i,j] = 255
            else:
                img1[i,j] = 0
    cv2.imshow('after intensity sliced', img1)
    cv2.waitKey(0)
    
def Contrast():
    img = cv2.imread('am5.jpeg')
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    cv2.imshow("after constrast", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def laplacian():
    kernel = np.array([[-1,-1,-1], 
                       [-1, 4,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(imgI, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
    sharpened = sharpened*82
    
    fin = cv2.addWeighted(imgI,0.5,sharpened,0.5,0)
    cv2.imshow("after",fin)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Sobel():
    kernel1 = np.array([[-1,-2,-1], 
                        [0,0,0],
                        [1,2,1]])
    kernel2 = np.array([[-1,0,1], 
                        [-2,0,2],
                        [-1,0,1]])
    
    f = cv2.filter2D(img, -1, kernel1) # applying the sharpening kernel to the input image & displaying it.
    n = cv2.filter2D(img, -1, kernel2)
    
    sharpen = cv2.addWeighted(f,0.5,n,0.5,0)
    #sharpen = sharpen +20
    fin = cv2.addWeighted(img,1,sharpen,0.5,0)

    #cv2.imwrite("sobel.jpg",fin)
    cv2.imshow("after sharped by sobel",fin)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Prewitt():
    kernel1 = np.array([[-1,0,1], 
                       [-1,0,1],
                       [-1,0,1]])
    kernel2 = np.array([[-1,-1,-1], 
                        [0,0,0],
                        [1,1,1]])

    f = cv2.filter2D(img, -1, kernel1) # applying the sharpening kernel to the input image & displaying it.
    n = cv2.filter2D(img, -1, kernel2)
    
    sharpen = cv2.addWeighted(f,0.5,n,0.5,0)
    #sharpen = sharpen *2
    fin = cv2.addWeighted(img,1,sharpen,0.5,0)
    
    #cv2.imwrite("sobel.jpg",fin)
    cv2.imshow("after",fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Mean():
    kernal = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernal)
    cv2.imshow("After Mean",dst)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Meadian():
    blur = cv2.medianBlur(img,5)
    cv2.imshow("After median",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Song():
    playsound('S.wav')
    
def volCtr():
    wCam,hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4,hCam)
    pTime = 0

    detector = htm.handDetector(detectionCon=0.7)



    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    #volume.GetMute()
    #volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)    
        if len(lmList)!=0:
            #print(lmList[4],lmList[8])
            
            x1,y1 = lmList[4][1],lmList[4][2]
            x2,y2 = lmList[8][1],lmList[8][2]
            cx,cy = (x1 + x2) // 2,(y1 + y2) // 2
            
            cv2.circle(img,(x1,y1),15,(87, 2, 16),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(87, 2, 16),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(87, 2, 16),3)
            cv2.circle(img,(cx,cy),15,(87, 2, 16),cv2.FILLED)
            
            length = math.hypot(x2 - x1,y2 - y1)
            #print(length)
            
            vol = np.interp(length,[50,300],[minVol,maxVol])
            volBar = np.interp(length,[50,300],[400,150])
            volPer = np.interp(length,[50,300],[0,100])
            #print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)
            
            if length<150:
                cv2.circle(img,(cx,cy),15,(255, 255, 102),cv2.FILLED)
                cv2.line(img,(x1,y1),(x2,y2),(255, 255, 102),3)
                cv2.circle(img,(x1,y1),15,(255, 255, 102),cv2.FILLED)
                cv2.circle(img,(x2,y2),15,(255, 255, 102),cv2.FILLED)
                
        cv2.rectangle(img, (50, 150),(85, 400), (0,0,0),5)
        cv2.rectangle(img, (51, int(volBar)),(85, 400), (0,0,255),cv2.FILLED)
        cv2.putText(img,f'{int(volPer)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,
                           1, (0,0,255),3)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
                
        cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,
                                        1, (255,0,0),3)
                
        cv2.imshow("img",img)
        cv2.waitKey(1)
                
#buttons
b1 = Button(root,text = 'Gussain',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = guassian).place(x=234,y=655)#
b2 = Button(root,text = 'Negative',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Negative).place(x=0,y=607)#
b3 = Button(root,text = 'Log',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 6,height=1,command = logTransmition).place(x=0,y=655)#
b4 = Button(root,text = 'power low',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = powerLow).place(x=84,y=655)#
b5 = Button(root,text = 'Blur',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = Blur).place(x=750,y=607)#
b6 = Button(root,text = 'identity',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = copy).place(x=684,y=655)#
b7 = Button(root,text = 'Max',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 6,height=1,command = MAX).place(x=0,y=559)
b8 = Button(root,text = 'Min',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = MIN).place(x=234,y=559)
b9 = Button(root,text = 'Smothing',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = SMOTH).place(x=384,y=655)#
b10 = Button(root,text = 'Histogram',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = Histogram).place(x=534,y=655)#
b11 = Button(root,text = 'intensity',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = intensity).place(x=834,y=655)#
b12 = Button(root,text = 'Contrast',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Contrast).place(x=984,y=655)#
b13 = Button(root,text = 'laplacian',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = laplacian).place(x=150,y=607)#
b14 = Button(root,text = 'Sobel',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Sobel).place(x=300,y=607)#
b15 = Button(root,text = 'Prewitt',bg = "#BA2A24",fg = "black",bd=4,font = 15,width = 12,height=1,command = Prewitt).place(x=450,y=607)#
b16 = Button(root,text = 'Mean',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Mean).place(x=600,y=607)#
b17 = Button(root,text = 'Meadian',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Meadian).place(x=84,y=559)#
b18 = Button(root,text = 'Enhance',bg = "#FFC3C3",fg = "black",bd=4,font = 15,width = 12,height=1,command = sharp).place(x=384,y=559)
#b17 = Button(root,text = 'volume',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Meadian).place(x=200,y=10)#
             
####################################################for insert photo in button########################################
photo = PhotoImage(file = r"D:\my work\2nd year\smester 2\imaging processing\sound3.png")
Button(root, text = 'Click Me !', image = photo,bg = "white",command = volCtr).place(x=1150,y=585)
######################################################################################################################

             
             
#b18 = Button(root,text = 'Song',bg = "#FFC3C3",fg = "black",bd=4,font = 15,width = 12,height=1,command = Song).place(x=534,y=559)

#T = tk.Label(root, text="team",bg = "#F6F4DB",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=25,y=10)
#G = tk.Label(root, text="Geegs  Salah",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=60)
#g = tk.Label(root, text="Jozeph Gamal",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=100)
#B = tk.Label(root, text="Brexam Gamal",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=140)

#b10 = Button(root,text = 'draw Rectangle',bg = "black",fg = "yellow",bd=4,font = 15,width = 12,height=1,command = Rectangle).place(x=12,y=442)
#b11 = Button(root,text = 'add Text',bg = "black",fg = "yellow",bd=4,font = 15,width = 12,height=1,command = Text).place(x=12,y=490)


#b2 = Button(root,text ='b2')
#b2.place(x=30,y=30)

#b3 = Button(root,text ='b2')

#show
root.mainloop()