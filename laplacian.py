import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Dental plaque 1 (2).jpg',0)

laplacin = cv2.Laplacian(img, cv2.CV_64F)

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('original'),plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacin,cmap='gray')
plt.title('laplacin'),plt.xticks([]), plt.yticks([])

# plt.imshow(plt, cmap="gray")
cv2.imshow("show ", laplacin)
cv2.waitKey(0)
cv2.destroyAllWindows()