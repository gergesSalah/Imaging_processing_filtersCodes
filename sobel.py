import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('15.jpg',0)

sobelx =cv2.sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely =cv2.sobel(img,cv2.CV_64F,0,1,ksize=5)


plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('original'),plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('sopel x'),plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('sopel y'),plt.xticks([]), plt.yticks([])

plt.show(img)

cv2.waitKey(0)
cv2.destroyAllWindows()