import cv2
import numpy as np

image = cv2.imread('0 After.jpg')
kernel = np.array([[-1,-1,-1], 
                       [-1, 9.5,-1],
                       [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.

cv2.imwrite("sharben in internet.jpg",sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()