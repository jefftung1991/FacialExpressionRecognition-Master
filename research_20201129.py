import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

os.environ["RUNFILES_DIR"] = "/Library/Frameworks/Python.framework/Versions/3.8/share/plaidml"
# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path

os.environ["PLAIDML_NATIVE_PATH"] = "/Library/Frameworks/Python.framework/Versions/3.8/lib/libplaidml.dylib"
# libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import cv2


labels = ['happy', 'others']

img_size = 224

if K.image_data_format() == 'channels_first':
  input_shape = (3, img_size, img_size)
else:
  input_shape = (img_size, img_size, 3)

def get_data(data_dir):
  data = [] 
  for label in labels: 
    path = os.path.join(data_dir, label)
    class_num = labels.index(label)
    for img in os.listdir(path):
      try:
        img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
        resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
        data.append([resized_arr, class_num])
      except Exception as e:
        print(e)
  return np.array(data)

train = get_data('train')
val = get_data('test')

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
  featurewise_center=False,  # set input mean to 0 over the dataset
  samplewise_center=False,  # set each sample mean to 0
  featurewise_std_normalization=False,  # divide inputs by std of the dataset
  samplewise_std_normalization=False,  # divide each input by its std
  zca_whitening=False,  # apply ZCA whitening
  rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
  zoom_range = 0.2, # Randomly zoom image 
  width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
  height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
  horizontal_flip = True,  # randomly flip images
  vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(64,(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256,(3,3), activation='relu'))
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

opt = Adam(lr=0.000001)

model.compile(optimizer = opt , loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) , metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs = 100 , validation_data = (x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# img_width, img_height = 48, 48
  
# train_data_dir = 'train'
# validation_data_dir = 'test'

# nb_train_samples = 493
# nb_validation_samples = 56
# epochs = 10
# batch_size = 64

# if K.image_data_format() == 'channels_first':
#   input_shape = (3, img_width, img_height)
# else:
#   input_shape = (img_width, img_height, 3)

# model = Sequential()
# model.add(Conv2D(64,(3,3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(128,(3,3), activation='relu'))
# model.add(Conv2D(128,(3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(256,(3,3), activation='relu'))
# model.add(Conv2D(256,(3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])


# train_datagen = ImageDataGenerator(
#     rescale = 1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size = batch_size,
#     class_mode = 'binary'
# )

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size = batch_size,
#     class_mode = 'binary'
# )

# history = model.fit(
#       train_generator,
#       steps_per_epoch=nb_train_samples // batch_size,
#       epochs=epochs,
#       validation_data=validation_generator,
#       validation_steps=nb_validation_samples // batch_size
#   )

# history_dict = history.history
# print(history_dict.keys())

# # plt.plot(history.history['accuracy']) 
# # plt.plot(history.history['val_accuracy'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'val'], loc='upper left')
# # plt.show()

# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('model loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'val'], loc='upper left')
# # plt.show()

# model.save_weights('first_try.h5')