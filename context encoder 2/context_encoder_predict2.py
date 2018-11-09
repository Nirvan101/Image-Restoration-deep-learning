from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from PIL import Image
from keras.callbacks import ModelCheckpoint
import random,numpy as np
import cv2,os
  

chkptPath = './weights/weights.hdf5'

'''
input_img = Input(shape=(256, 384, 3))
x = Conv2D(64, (4, 4), strides = 2, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 3),strides=(2,3), padding='same')(x)
x = Flatten()(x)
x = Dense(256 * 4 * 4 ,activation='relu')(x)

x = Reshape((4,4,256))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 3))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
'''

#new architecture 2
input_img = Input(shape=(256, 384, 3))
x = Conv2D(64, (4, 4), strides = 2, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 2), padding='same')(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(256 * 16 * 12 ,activation='relu')(x)

x = Reshape((16,12,256))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((1, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


model = Model(input_img, x)
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='sgd')
#model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()


#----------------------------------------------------------
#predict
#----------------------------------------------------------

model.load_weights(chkptPath)

#train images
'''
test_path2 = './dataset/good_pictures'
save_path = './train_resized'
save_path2 = './train_output'

for i in os.listdir(test_path2 + '/lol'):
    #predict
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = cv2.resize(img,(384,256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    
    for a in res:
        a *= 255.0
        im = Image.fromarray(a.astype('uint8'))
        im.save(save_path2 + '/out_' + i)
    
    
    #save original image
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = np.array(img)  #/ 255.0
    img = cv2.resize(img,(384,246))
    im = Image.fromarray(img.astype('uint8'))
    im.save(save_path + '/' + i)
'''

#test images
test_path2 = './one'
save_path = './test_resized'
save_path2 = './test_output'


for i in os.listdir(test_path2 + '/lol'):
    #predict
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = cv2.resize(img,(384,256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    
    for a in res:
        a *= 255.0
        im = Image.fromarray(a.astype('uint8'))
        im.save(save_path2 + '/out_' + i)
    
    
    #save original image
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = np.array(img)  #/ 255.0
    img = cv2.resize(img,(384,246))
    im = Image.fromarray(img.astype('uint8'))
    im.save(save_path + '/' + i)
     
  
        
