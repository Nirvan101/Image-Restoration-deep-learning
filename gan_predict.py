from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from PIL import Image
from keras.callbacks import ModelCheckpoint
import random,numpy as np
import cv2

def cropper(batch):
    res = []
    for im in batch:
        h,w,_ = im.shape
        im2 = im.copy()
        for i in range(500):
            x = random.randint(1,w-1)
            y = random.randint(1,h-1)
            l = random.choice( [2,1,3,5] )
            x2 = x+l
            y2 = y+l
            cv2.rectangle(im2, (x,y) , ( x2,y2 ) ,(255,255,255),-1)
    
        res.append(np.array(im2))
    res = np.array(res)
#    res /= 255.0
#    batch /= 255.0
    return res


def myloss(y_true,y_pred):
    diff = (y_pred-y_true)  
    diff = diff[ 96:160 , 144:240 ]
    return 0.5*K.mean(K.square(diff))
    

source_path = './dataset/good_pictures'
validation_path = './dataset/validation'



#generator
a1 = Input(shape=(256, 384, 3))
a2 = Conv2D(64, (4, 4), strides = 2, activation='relu', padding='same')(a1)
x = MaxPooling2D((2, 2), padding='same')(a2)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 2), padding='same')(x)
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
outG = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

gen = Model(a1, outG)
gen.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
gen.summary()

#discriminator
b1 = Input(shape=(256, 384, 3))
b2 = Conv2D(32, (4, 4), strides = 2, activation='relu', padding='same')(b1)
x = MaxPooling2D((2, 2), padding='same')(b2)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((3, 3), padding='same')(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(64 ,activation='relu')(x)
outD = Dense(1 ,activation='relu')(x)

disc = Model(b1,outD)
disc.summary()
disc.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])


disc.trainable = False


#combined model
combined = Model( gen.input , [gen.output, disc(gen.output)])
combined.summary()
combined.compile(loss=['mse', 'binary_crossentropy'],
            optimizer='adam',
            metrics=['accuracy'])


#----------------------------------------------------------
#predict
#----------------------------------------------------------


gen_weights_path = './gan_weights/gen/weights.hdf5'
comb_weights_path = './gan_weights/comb/weights.hdf5'
disc_weights_path = './gan_weights/disc/weights.hdf5'


gen.load_weights(gen_weights_path)

source_path = './dataset/good_pictures'
save_path = './predictions/two'

predgen = ImageDataGenerator(rescale = 1./255)
test_images = predgen.flow_from_directory(source_path,class_mode=None,target_size = (256,384),
                                          batch_size=32)
arr = gen.predict_generator(test_images,steps = 1)


i=1
for a in arr:
    print(a.shape)
    a *= 255
    a = a[96:160 , 144:240 , :]
    im = Image.fromarray(a.astype('uint8'))
    im.save(save_path + '/out' + str(i) + '.jpeg')
    i += 1



                           
