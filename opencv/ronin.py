import cv2
import numpy as np
import os

image = cv2.imread("panda.png")

#grayImage = cv2.imread("panda.png", cv2.CV_LOAD_IMAGE_GRAYSCALE )
#cv2.imshow(image,"window")

imgByteArr = bytearray(image)
randomByteArr = bytearray(os.urandom(120000))
print(randomByteArr)
flatNumpyArr = np.array(randomByteArr)
print(flatNumpyArr)
grayImage = flatNumpyArr.reshape(300,400)
cv2.imwrite("Random_Gray.png",grayImage)

bgrImage = flatNumpyArr.reshape(100,400,3)
cv2.imwrite("Random_Color.png", bgrImage)

#cv2.imwrite("duplicate_panda.jpg", image)
