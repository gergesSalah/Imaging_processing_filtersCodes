import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 # Enter the image
image = cv2.imread('15.')
lena = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
 # Gray conversion processing
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#kernel
kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype= int)
kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],dtype= int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelX)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernelY)
 
 #   uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
 
 #  
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
 
 # Image display
plt.rcParams['font.sans-serif'] = ['SimHei']
 Titles = [u 'Original Image', U 'Prewitt Image']
images = [lena, Prewitt]
 
for i in range(2): 
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
plt.show()