# install pnslib
!pip install git+git://github.com/PnS2019/pnslib.git

# check opencv version
import cv2
from pnslib import utils
# example of face detection with opencv cascade classifier
from cv2 import imread
from cv2 import CascadeClassifier
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# load the pre-trained model
face_cascade = cv2.CascadeClassifier(
    utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
# eye_cascade = cv2.CascadeClassifier(
#     utils.get_haarcascade_path('haarcascade_eye.xml'))

# load the photograph
# image_path = "drive/My Drive/jeff.jpg"
image_path = "5a4bdcff390942869a5b4ad52e7e5e5d-1.jpg"
img = imread(image_path)

# perform face detection
# search face
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
plt.show()