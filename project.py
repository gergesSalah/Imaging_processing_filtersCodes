import tkinter
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageFilter
import tkinter as tk

from matplotlib import pyplot as plt
from playsound import playsound
import os
import time
# import HandTracking as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# song
# Everything stops when played
from sklearn.neighbors import KNeighborsClassifier

os.system("Sam21.wav")

# Everything continues when played
os.system("Sam21.wav&")

# frame
root = Tk()
root.title('Imaging processing')
root.geometry("1218x700")
root.resizable(width=False, height=False)
C = Canvas(root, bg='#595959')
C.pack(fill=BOTH, expand=1)

# img
image1 = Image.open("still-life-sustainable-lifestyle-elements-arrangement_23-2148997049.jpg")
image1 = image1.resize((1218, 700), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(image=test)
label1.image = test

# Position image
label1.place(x=-10, y=0)

# #design
# image1 = Image.open("W3.jpg")
# test = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test)
# label1.image = test
# label1.place(x=0, y=558)
#
# image1 = Image.open("W3.jpg")
# test = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test)
# label1.image = test
# label1.place(x=167, y=558)
#
# image1 = Image.open("W3.jpg")
# test = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test)
# label1.image = test
# label1.place(x=524, y=606)
#
# image1 = Image.open("W3.jpg")
# test = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test)
# label1.image = test
# label1.place(x=533, y=606)
#
# image1 = Image.open("W3.jpg")
# test = ImageTk.PhotoImage(image1)
# label1 = tkinter.Label(image=test)
# label1.image = test
# label1.place(x=766, y=654)


# the image ofconstract function
# in functioin scope
# img for max and min fliter
imgO = Image.open("am5.jpeg")
# global variable for functions
img = cv2.imread('am5.jpeg')
# for intensity function
imgI = cv2.imread('am5.jpeg', 0)


# functions
def Bimodal():
    # Load a grayscale image (or create a binary image)
    image = np.zeros((10, 10), dtype=np.uint8)
    image[3:7, 3:7] = 1  # A square in the center

    # Otsu's Bimodal Thresholding
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the original image and the thresholded image
    titles = ["Original Image", "Bimodal Thresholded Image"]
    images = [image, thresholded]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def LaplacOfGussian():
    img = cv2.imread('image.jpg', 0)

    laplacin = cv2.Laplacian(img, cv2.CV_64F)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacin, cmap='gray')
    plt.title('laplacin'), plt.xticks([]), plt.yticks([])

    # plt.imshow(plt, cmap="gray")
    cv2.imshow("show ", laplacin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Hit_Fit():
    # Create a binary image (or load one)
    image = np.zeros((10, 10), dtype=np.uint8)
    image[3:7, 3:7] = 1  # A square in the center

    # Define structuring elements
    # For "Hit" operation (foreground)
    structuring_element_foreground = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)

    # For "Fit" operation (background)
    structuring_element_background = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ], dtype=np.uint8)

    # Perform "Hit" (foreground erosion)
    hit = cv2.erode(image, structuring_element_foreground, iterations=1)

    # Perform "Fit" (background erosion on the complement of the image)
    image_complement = cv2.bitwise_not(image) // 255  # Convert complement to binary
    fit = cv2.erode(image_complement, structuring_element_background, iterations=1)

    # Display the results
    titles = ["Original Image", "Hit Operation", "Fit Operation"]
    images = [image, hit, fit]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def AdabThreshold():
    # Load the image
    image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or cannot be loaded.")

    # Apply adaptive thresholding
    # Mean Adaptive Thresholding
    mean_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Gaussian Adaptive Thresholding
    gaussian_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Display the results
    titles = ['Original Image', 'Mean Adaptive Threshold', 'Gaussian Adaptive Threshold']
    images = [image, mean_thresh, gaussian_thresh]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def Blur():
    dts = cv2.blur(img, (9, 9))
    cv2.imshow("After", dts)
    cv2.waitKey()
    cv2.destroyAllWindows()


def k_means():
    # Load the image
    image = cv2.imread("kmeanImage.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for clustering

    # Reshape the image into a 2D array of pixels (rows: pixels, cols: features)
    pixel_values = image_gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)  # Convert to float32 for k-means

    # Define criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of clusters (K) - e.g., 2 for binary segmentation
    k = 2
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the labels back to an image
    labels = labels.flatten()
    segmented_image = labels.reshape(image_gray.shape)

    # Map each cluster to its intensity (center values)
    segmented_image = np.uint8(segmented_image * (255 // (k - 1)))

    # Display the original and segmented images
    titles = ["Original Image", "Segmented Image"]
    images = [image_rgb, segmented_image]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 1 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


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


def GlobalThresh():
    # Load the image
    image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or cannot be loaded.")

    # Apply global thresholding
    # The threshold value is set to 127, and maximum value to 255
    _, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, binary_inv_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    _, trunc_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    _, tozero_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    _, tozero_inv_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

    # Display the results
    titles = [
        "Original Image",
        "Binary Threshold",
        "Binary Inv Threshold",
        "Trunc Threshold",
        "ToZero Threshold",
        "ToZero Inv Threshold"
    ]
    images = [image, binary_thresh, binary_inv_thresh, trunc_thresh, tozero_thresh, tozero_inv_thresh]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# def Rectangle():
#   startpoint = (30,30)
#  endpoint = (60,60)
# col = (255,0,0)
#  thick = 2
#    newimage = cv2.rectangle(img,startpoint,endpoint,col,thick)
#   cv2.imshow("rectangle",newimage)
#  cv2.waitKey(0)
# cv2.destroyAllWindows()

# def Text():
#   org = (50,50)
#  font = cv2.FONT_HERSHEY_COMPLEX
# fontScale = 2
# color = (40,255,0)
#  thick = 2
#    newImage = cv2.putText(img,"-----",org,font,fontScale,color,thick)
#   cv2.imshow("after add Text",newImage)
#  cv2.waitKey(0)
# cv2.destroyAllWindows()

def sharp():
    dst = cv2.blur(img, (9, 9))

    sub = cv2.subtract(img, dst)

    row, col, chan = sub.shape
    # maltiplay this sup by num
    subFac = np.zeros(sub.shape, np.uint8)
    for i in range(row):
        for j in range(col):
            subFac[i, j] = sub[i, j] * 3

    fin = cv2.addWeighted(subFac, 0.5, img, 0.9, 0)

    messagebox.showinfo('information', "this function is not required but we made it by self")
    cv2.imshow("after enhancment", fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ErosionDilt():
    # Create a binary image
    image = np.zeros((10, 10), dtype=np.uint8)
    image[3:7, 3:7] = 1  # A square in the center
    # Structuring element (kernel)
    kernel = np.ones((3, 3), dtype=np.uint8)  # 3x3 square structuring element

    # Perform erosion
    erosion = cv2.erode(image, kernel, iterations=1)

    # Perform dilation
    dilation = cv2.dilate(image, kernel, iterations=1)

    # Display the original, eroded, and dilated images
    titles = ["Original Image", "Erosion", "Dilation"]
    images = [image, erosion, dilation]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def KNN():
    # Load the image
    image = cv2.imread("KnnImage.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for simplicity

    # Reshape the image into a 2D array of pixels and features
    pixel_values = image_gray.reshape((-1, 1))

    # Generate labels for clustering (e.g., manually labeled or simulated for KNN)
    # For simplicity, simulate labels using random sampling
    num_pixels = len(pixel_values)
    labels = np.random.choice([0, 1], size=num_pixels, replace=True)  # Binary labels (0 or 1)

    # Split into training and testing sets
    split_ratio = 0.8
    train_size = int(split_ratio * num_pixels)
    train_pixels = pixel_values[:train_size]
    train_labels = labels[:train_size]
    test_pixels = pixel_values[train_size:]
    test_labels = labels[train_size:]

    # Train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)  # k = 3
    knn.fit(train_pixels, train_labels)

    # Predict the labels for the test set
    predicted_labels = knn.predict(pixel_values)

    # Reshape the predicted labels to match the original image dimensions
    segmented_image = predicted_labels.reshape(image_gray.shape)

    # Display the original and segmented images
    titles = ["Original Image", "Segmented Image"]
    images = [image_rgb, segmented_image]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 1 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def laplacian():
    img = cv2.imread('image.jpg', 0)

    laplacin = cv2.Laplacian(img, cv2.CV_64F)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacin, cmap='gray')
    plt.title('laplacin'), plt.xticks([]), plt.yticks([])

    # plt.imshow(plt, cmap="gray")
    cv2.imshow("show ", laplacin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def laplacian():
    kernel = np.array([[-1, -1, -1],
                       [-1, 4, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(imgI, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
    sharpened = sharpened * 82

    fin = cv2.addWeighted(imgI, 0.5, sharpened, 0.5, 0)
    cv2.imshow("after", fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Sobel():
    kernel1 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    kernel2 = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    f = cv2.filter2D(img, -1, kernel1)  # applying the sharpening kernel to the input image & displaying it.
    n = cv2.filter2D(img, -1, kernel2)

    sharpen = cv2.addWeighted(f, 0.5, n, 0.5, 0)
    # sharpen = sharpen +20
    fin = cv2.addWeighted(img, 1, sharpen, 0.5, 0)

    # cv2.imwrite("sobel.jpg",fin)
    cv2.imshow("after sharped by sobel", fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Prewitt():
    kernel1 = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    kernel2 = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])

    f = cv2.filter2D(img, -1, kernel1)  # applying the sharpening kernel to the input image & displaying it.
    n = cv2.filter2D(img, -1, kernel2)

    sharpen = cv2.addWeighted(f, 0.5, n, 0.5, 0)
    # sharpen = sharpen *2
    fin = cv2.addWeighted(img, 1, sharpen, 0.5, 0)

    # cv2.imwrite("sobel.jpg",fin)
    cv2.imshow("after", fin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Mean():
    kernal = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernal)
    cv2.imshow("After Mean", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Meadian():
    blur = cv2.medianBlur(img, 5)
    cv2.imshow("After median", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Song():
    playsound('S.wav')


#
# def volCtr():
#     wCam,hCam = 640, 480
#     cap = cv2.VideoCapture(0)
#     cap.set(3,wCam)
#     cap.set(4,hCam)
#     pTime = 0
#
#     detector = htm.handDetector(detectionCon=0.7)
#
#
#
#     devices = AudioUtilities.GetSpeakers()
#     interface = devices.Activate(
#             IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#     volume = cast(interface, POINTER(IAudioEndpointVolume))
#     #volume.GetMute()
#     #volume.GetMasterVolumeLevel()
#     volRange = volume.GetVolumeRange()
#     minVol = volRange[0]
#     maxVol = volRange[1]
#     vol = 0
#     volBar = 400
#     volPer = 0
#
#     while True:
#         success,img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img, draw = False)
#         if len(lmList)!=0:
#             #print(lmList[4],lmList[8])
#
#             x1,y1 = lmList[4][1],lmList[4][2]
#             x2,y2 = lmList[8][1],lmList[8][2]
#             cx,cy = (x1 + x2) // 2,(y1 + y2) // 2
#
#             cv2.circle(img,(x1,y1),15,(87, 2, 16),cv2.FILLED)
#             cv2.circle(img,(x2,y2),15,(87, 2, 16),cv2.FILLED)
#             cv2.line(img,(x1,y1),(x2,y2),(87, 2, 16),3)
#             cv2.circle(img,(cx,cy),15,(87, 2, 16),cv2.FILLED)
#
#             length = math.hypot(x2 - x1,y2 - y1)
#             #print(length)
#
#             vol = np.interp(length,[50,300],[minVol,maxVol])
#             volBar = np.interp(length,[50,300],[400,150])
#             volPer = np.interp(length,[50,300],[0,100])
#             #print(int(length), vol)
#             volume.SetMasterVolumeLevel(vol, None)
#
#             if length<150:
#                 cv2.circle(img,(cx,cy),15,(255, 255, 102),cv2.FILLED)
#                 cv2.line(img,(x1,y1),(x2,y2),(255, 255, 102),3)
#                 cv2.circle(img,(x1,y1),15,(255, 255, 102),cv2.FILLED)
#                 cv2.circle(img,(x2,y2),15,(255, 255, 102),cv2.FILLED)
#
#         cv2.rectangle(img, (50, 150),(85, 400), (0,0,0),5)
#         cv2.rectangle(img, (51, int(volBar)),(85, 400), (0,0,255),cv2.FILLED)
#         cv2.putText(img,f'{int(volPer)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,
#                            1, (0,0,255),3)
#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
#
#         cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,
#                                         1, (255,0,0),3)
#
#         cv2.imshow("img",img)
#         cv2.waitKey(1)
#
# buttons
b1 = Button(root, text='Bimodal thresh', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1, command=Bimodal).place(
    x=234, y=655)  #
b2 = Button(root, text='Gussian', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=LaplacOfGussian).place(
    x=0, y=607)  #
b3 = Button(root, text='hit-fit', bg="#BA2A24", fg="black", bd=4, font=15, width=6, height=1, command=Hit_Fit).place(
    x=0, y=655)  #
b4 = Button(root, text='Adbt threshold', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1,
            command=AdabThreshold).place(x=84, y=655)  #
b5 = Button(root, text='Blur', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1, command=Blur).place(x=750,
                                                                                                                y=607)  #
b6 = Button(root, text='k-mean', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=k_means).place(
    x=684, y=655)  #
b7 = Button(root, text='Max', bg="#BA2A24", fg="black", bd=4, font=15, width=6, height=1, command=MAX).place(x=0, y=559)
b8 = Button(root, text='Min', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1, command=MIN).place(x=234,
                                                                                                              y=559)
b9 = Button(root, text='global thresh', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=GlobalThresh).place(
    x=384, y=655)  #
b10 = Button(root, text='ErosionDilition', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1,
             command=ErosionDilt).place(x=534, y=655)  #
b11 = Button(root, text='KNN', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1,
             command=KNN).place(x=834, y=655)  #
b12 = Button(root, text='laplacian', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1,
             command=laplacian).place(x=984, y=655)  #
b13 = Button(root, text='laplacian', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1,
             command=laplacian).place(x=150, y=607)  #
b14 = Button(root, text='Sobel', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=Sobel).place(
    x=300, y=607)  #
b15 = Button(root, text='Prewitt', bg="#BA2A24", fg="black", bd=4, font=15, width=12, height=1, command=Prewitt).place(
    x=450, y=607)  #
b16 = Button(root, text='Mean', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=Mean).place(x=600,
                                                                                                                 y=607)  #
b17 = Button(root, text='Meadian', bg="#9B1913", fg="black", bd=4, font=15, width=12, height=1, command=Meadian).place(
    x=84, y=559)  #
b18 = Button(root, text='Enhance', bg="#FFC3C3", fg="black", bd=4, font=15, width=12, height=1, command=sharp).place(
    x=384, y=559)
# b17 = Button(root,text = 'volume',bg = "#9B1913",fg = "black",bd=4,font = 15,width = 12,height=1,command = Meadian).place(x=200,y=10)#

####################################################for insert photo in button########################################
photo = PhotoImage(file=r"D:\myWork\2nd year\smester 2\imaging processing\sound3.png")
# Button(root, text = 'Click Me !', image = photo,bg = "white",command = volCtr).place(x=1150,y=585)
######################################################################################################################


# b18 = Button(root,text = 'Song',bg = "#FFC3C3",fg = "black",bd=4,font = 15,width = 12,height=1,command = Song).place(x=534,y=559)

# T = tk.Label(root, text="team",bg = "#F6F4DB",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=25,y=10)
# G = tk.Label(root, text="Geegs  Salah",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=60)
# g = tk.Label(root, text="Jozeph Gamal",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=100)
# B = tk.Label(root, text="Brexam Gamal",bg = "#595959",fg = "black",bd=4,font = 15,width = 12,height=1).place(x=18,y=140)

# b10 = Button(root,text = 'draw Rectangle',bg = "black",fg = "yellow",bd=4,font = 15,width = 12,height=1,command = Rectangle).place(x=12,y=442)
# b11 = Button(root,text = 'add Text',bg = "black",fg = "yellow",bd=4,font = 15,width = 12,height=1,command = Text).place(x=12,y=490)


# b2 = Button(root,text ='b2')
# b2.place(x=30,y=30)

# b3 = Button(root,text ='b2')

# show
root.mainloop()