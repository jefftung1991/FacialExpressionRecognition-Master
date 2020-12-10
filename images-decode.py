import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image

fer_data=pd.read_csv('fer2013.csv',delimiter=',')

def save_fer_img():

    for index,row in fer_data.iterrows():
        pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        img=pixels.reshape((48,48))

        if(row['Usage'] == 'PrivateTest'):
            root_folder = 'Test'
        else:
            root_folder = 'Train'

        if (row['emotion'] == 3):
            pathname=os.path.join(root_folder,'Happy', 'Happy_fer_images_' + str(index)+'.jpg')
        else:
            pathname=os.path.join(root_folder,'Others','Other_fer_images_' + str(index)+'.jpg')
        
        # cv2.imwrite(pathname,img)
        im = Image.fromarray(img)
        im.save(pathname)
        print('image saved ias {}'.format(pathname))

save_fer_img()