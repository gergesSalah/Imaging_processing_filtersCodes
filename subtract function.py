import cv2

img1= cv2.imread("S.jpg")
img2 = cv2.imread("b.jpg")

subtracted = cv2.subtract(img1,img2)

cv2.imshow("subtracted image",subtracted)


cv2.waitKey(0)
cv2.destroyAllWindows()