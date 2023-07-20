import cv2
import  numpy as np

img = cv2.imread("images33.jpg")
print(img.shape)

row,cols,chan = img.shape
print(row)

img2 = np.zeros((row,cols,chan),np.uint8)
for i in range(row):
    for j in range(cols):
        img2[i,j] = 255 - img[i,j]
        
cv2.imshow("img",img2)
cv2.imwrite("new negative photo.jpg",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()