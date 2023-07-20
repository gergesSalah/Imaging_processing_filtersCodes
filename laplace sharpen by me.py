import cv2
import numpy as np

img = cv2.imread('Dental plaque 1 (2).jpg')
kernel = np.array([[-1,-1,-1], 
                   [-1, 4,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel) # applying the sharpening kernel to the input image & displaying it.

fin = cv2.addWeighted(img,0.9,sharpened,0.5,0)

cv2.imwrite("laplace sharp.jpg",fin)
cv2.imshow("after",fin)
cv2.imshow("before",img)

cv2.waitKey(0)
cv2.destroyAllWindows()