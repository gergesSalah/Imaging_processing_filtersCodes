import cv2
import os

img1 = cv2.imread("S.jpg")
img2 = cv2.imread("b.jpg")
weightedimg = cv2.addWeighted(img1,0.8,img2,0.3,0)
cv2.imshow("weighted image",weightedimg)

directory = r"D:\myWork\2nd year\smester 2\imaging processing"

os.chdir(directory)
cv2.imwrite("addWeighted.jpg",weightedimg)
print(os.listdir(directory))

cv2.waitKey(0)
cv2.destroyAllWindows()