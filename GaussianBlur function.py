import cv2
import numpy as np

img = cv2.imread('0.jpg')

dts = cv2.GaussianBlur(img,(35,35),40)

cv2.imshow("before",img)
cv2.imshow("After",dts)

cv2.imwrite("0 after with gaussi.jpg",dts)

cv2.waitKey()
cv2.destroyAllWindows()