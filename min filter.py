import cv2
from PIL import Image,ImageFilter

img = Image.open("s3 with noise.jpg")

dts = img.filter(ImageFilter.MinFilter(size=3))

dts.show()

cv2.waitKey()
cv2.destroyAllWindows()