import cv2

camp = cv2.VideoCapture("L.mp4")

while(camp.isOpened()):
    rtn, img = camp.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",img)
    cv2.imwrite("Lgerges.jpg",img)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break

cv2.release()
cv2.destroyAllWindows()