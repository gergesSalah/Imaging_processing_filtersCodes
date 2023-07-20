import cv2
import numpy as np

img = cv2.imread("s.jpg")

row,col,chan = img.shape

img2 = np.zeros(img.shape,np.uint8)

for i in range(row):
    for j in range(col):
        img2[i,j] = img[i,j]

cv2.imshow("img",img2)
cv2.imwrite("copy image.jpg",img2)

cv2.waitKey()
cv2.destroyAllWindows()
 