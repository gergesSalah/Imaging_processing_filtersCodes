import cv2
import numpy as np

img = cv2.imread('s3.jpg')

dts = cv2.blur(img,(9,9))

cv2.imshow("before",img)
cv2.imshow("After",dts)

cv2.waitKey()
cv2.destroyAllWindows()