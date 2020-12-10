from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


filepath =  'Asian_First_Try.h5'

class_list = ['Happy', 'Others']
# dimensions of our images
img_width, img_height = 48, 48

# Load the model
model = load_model(filepath)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# summarize input and output shape
print(model.inputs)
print(model.outputs)

# predicting images
img = image.load_img('asian_angry.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (class_list[classes.astype(int)[0][0]])


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# orig = cv2.imread('therocksad2.jpg')
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (64, 64))


# faces = face_cascade.detectMultiScale(image, 1.3, 5)

# image = image.astype("float") / 255.0

# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)



# for (x,y,w,h) in faces:
    
#     sub_face = image[y:y+h, x:x+w]

#     # make predictions on the input image
#     pred = model.predict(sub_face)
#     pred = pred.argmax(axis=1)[0]
#     print (pred)
    
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # font = cv2.FONT_HERSHEY_DUPLEX
    
    # dim = (img_width, img_height)
    # # resize image
    # resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

    # x = image.img_to_array(resized)
    # x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    # classes = np.argmax(model.predict(images), axis=-1)
    # print (classes)


    # cv2.putText(img,
    #                 class_list[classes.astype(int)[0]],
    #                 # "test",
    #                 (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (0, 0, 255), 2)
    

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
