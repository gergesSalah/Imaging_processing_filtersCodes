import cv2
import os

imgpath = r"D:\my work\2nd year\smester 2\imaging processing\b.jpg"
directory = r"D:\my work\2nd year\smester 2\imaging processing\newimg"

src = cv2.imread(imgpath)

org = (455,310)

font = cv2.FONT_HERSHEY_COMPLEX

fontScale = 5

color = (200,255,0)

thick = 2


newImage = cv2.putText(src,'gerges',org,font,fontScale,color,thick)

os.chdir(directory)
cv2.imwrite("gergesText.jpg",newImage)

cv2.imshow("newText",newImage)


cv2.waitKey(0)
cv2.destroyAllWindows()