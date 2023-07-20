import cv2
import numpy as np

img = cv2.imread("Dental plaque 1 (2).jpg")
cv2.imshow("show image ", img)
row,col,chan = img.shape

img2 = np.zeros(img.shape,np.uint8)

for i in range(row):
    for j in range(col):
        img2[i,j] = 85 * np.log(img[i,j])

cv2.imshow("img",img2)
cv2.imwrite("log transmitted photo.jpg",img2)

cv2.waitKey()
cv2.destroyAllWindows()
 