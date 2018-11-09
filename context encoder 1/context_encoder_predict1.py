from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from PIL import Image
from keras.callbacks import ModelCheckpoint
import random,numpy as np
import cv2
  

chkptPath = './weights/weights.hdf5'
test_path = './one'
test_path2 = './dataset/good_pictures'
save_path = './two_predict_on_test'
save_path2 = './two_predict_on_train'
save_path3 = './predictions/one/train'

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

def fixer(batches):
    for batch in batches:
        batch /= 255.0
        yield batch
        
def fixer2(img):
    img = np.array(img)
    img /= 255.0
    return img
        

model.load_weights(chkptPath)

predgen = ImageDataGenerator(rescale = 1./255)
test_images = predgen.flow_from_directory(test_path2,class_mode=None,target_size = (256,384),
                                          batch_size=32)
arr = model.predict_generator(test_images,steps=1)

'''
im = Image.fromarray(arr[0].astype('uint8'))
im.save(save_path + '/out' + '.jpeg')
'''
print('HERE.')

i=1
for a in arr:
    print(a.shape)
    a *= 255
    im = Image.fromarray(a.astype('uint8'))
    im.save(save_path3 + '/out' + str(i) + '.jpeg')
    i += 1




       
  
        
