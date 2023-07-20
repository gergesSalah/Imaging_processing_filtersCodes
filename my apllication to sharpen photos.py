import cv2
import numpy as np
img = cv2.imread("1 (1).jpg")

dst = cv2.blur(img,(9,9))

sub = cv2.subtract(img,dst)

row,col,chan = sub.shape
#maltiplay this sup by num
subFac = np.zeros(sub.shape,np.uint8)
for i in range(row):
    for j in range(col):
        subFac[i,j] = sub[i,j]*7

fin = cv2.addWeighted(subFac,0.5,img,0.7,0)

cv2.imwrite("finish 25.jpg",fin)

cv2.imshow("before",img)
cv2.imshow("after",fin)

cv2.waitKey(0)
cv2.destroyAllWindows()