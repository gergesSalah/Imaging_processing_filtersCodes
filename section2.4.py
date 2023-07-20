import cv2
import os

imgpath = r"D:\my work\2nd year\smester 2\imaging processing\g.jpg"

directory = r"D:\my work\2nd year\smester 2\imaging processing\newimg"
src = cv2.imread(imgpath)

startpoint = (800,500)
endpoint = (900,600)

col = (255,0,0)

thick = 100

newimage = cv2.rectangle(src,startpoint,endpoint,col,thick)
cv2.imshow("rectangle",newimage)

os.chdir(directory)
cv2.imwrite("gerges2.jpg",newimage)

cv2.waitKey(0)
cv2.destroyAllWindows()