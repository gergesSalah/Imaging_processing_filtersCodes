import cv2
import numpy as np

img = cv2.imread('s3 with noise.jpg')

dts = cv2.medianBlur(img,7)

cv2.imwrite("qq.jpg",dts)

cv2.imshow("before",img)
cv2.imshow("After",dts)


cv2.waitKey()
cv2.destroyAllWindows()
