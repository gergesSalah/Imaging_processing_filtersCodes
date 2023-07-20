import cv2

img = cv2.imread("g.jpg",0) #cv2.IMREAD_COLOR

cv2.imshow("view",img)

cv2.waitKey(0)

cv2.destroyAllWindows()

