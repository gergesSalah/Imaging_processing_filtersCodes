import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gerges.jpg",gray)#do it download photo every second but because all of them has the same name so python replaced it by each other
    cv2.imshow("frame",gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):#what is meaning of 0XFF
        break;

cv2.release()
cv2.destroyAllWindows()