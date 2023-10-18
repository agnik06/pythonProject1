import cv2
import cv2 as cv
import numpy as np
import numpy as num

img = cv.imread('group-of-people-18.jpg ')
resize=cv.resize(img,(900,900))

harr = cv.CascadeClassifier('eyeglass.xml')
faces = harr.detectMultiScale(resize, scaleFactor=1.1, minNeighbors=4)
for (x, y, z, w) in faces:
    cv.rectangle(resize, (x, y), (x + w, y + z), (0, 250, 0 ),  thickness=2)
cv.imshow('face', resize)
cv.waitKey(0)
