import cv2

img1 = cv2.imread("g.jpg")
img2 = cv2.imread("b.jpg")

newImg = cv2.addWeighted(img1,img2)

cv2.imshow("new",newImg)

cv2.waitKey(0)
cv2.destroyAllWindows()