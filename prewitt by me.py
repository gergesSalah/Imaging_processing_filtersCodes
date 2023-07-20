import cv2
import numpy as np

img = cv2.imread('l1.jpg')
kernel1 = np.array([[-1,0,1], 
                    [-1,0,1],
                    [-1,0,1]])
kernel2 = np.array([[-1,-1,-1], 
                    [0,0,0],
                    [1,1,1]])

f = cv2.filter2D(img, -1, kernel1) # applying the sharpening kernel to the input image & displaying it.
n = cv2.filter2D(img, -1, kernel2)

sharpen = cv2.addWeighted(f,0.5,n,0.5,0)
sharpen = sharpen *2
fin = cv2.addWeighted(img,1,sharpen,0.5,0)

#cv2.imwrite("sobel.jpg",fin)
cv2.imshow("after",fin)
cv2.imshow("before",img)

cv2.waitKey(0)
cv2.destroyAllWindows()