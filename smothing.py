import cv2
import numpy as np

img = cv2.imread('s3.jpg')

karnel = np.ones((9,9),np.float32)/25 #why we devision on 25 , and what is float32

dts = cv2.filter2D(img,-1,karnel)

cv2.imshow("before",img)
cv2.imshow("After",dts)

cv2.waitKey()
cv2.destroyAllWindows()