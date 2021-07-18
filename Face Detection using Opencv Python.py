import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

#Loading the image to be tested
test_image = cv2.imread('data/1.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying the grayscale image
plt.imshow(test_image_gray, cmap='gray')

# Since we know that OpenCV loads an image in BGR format, so we need to convert it into RBG format to be able to display its true colors. Let us write a small function for that.
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Loading the Classifier for frontal face
    
haar_cascade_face = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')
 #face detection

 faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))

Faces found:  1

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#convert image to RGB and show image

plt.imshow(convertToRGB(test_image))