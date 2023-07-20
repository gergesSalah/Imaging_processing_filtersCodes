import cv2
import numpy as np
def nothing():
    pass
#creat black image
img = np.zeros((512,512,3), np.uint8)# why 3 : the r g b
cv2.namedWindow('image')
#creat track bars for change colour
cv2.createTrackbar('R','image',0,255,nothing)#why 'image'parameter
cv2.createTrackbar('G','image',0,255,nothing)#why 'image'parameter
cv2.createTrackbar('B','image',0,255,nothing)#why 'image'parameter
#creat switch for on/off 
switch = '0 : OFF\n1 : ON'# try to delete this 
cv2.createTrackbar(switch,'image',0,1,nothing)#try replace the 1 with 0
while(1):
    cv2.imshow('image' , img)
    k = cv2.waitKey(1) & 0xFF #what this line do
    if k==27:
        break
    # get current possitions
    r = cv2.getTrackbarPos('R','image')#why image
    g = cv2.getTrackbarPos('G','image')#why image
    b = cv2.getTrackbarPos('B','image')#why image
    s = cv2.getTrackbarPos(switch,'image')#why image
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
cv2.destroyAllWindows()
        
