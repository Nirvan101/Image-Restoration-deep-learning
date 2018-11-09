from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from PIL import Image
from keras.callbacks import ModelCheckpoint
import random,numpy as np
import cv2
import tensorflow as tf

source_path = './dataset/good_pictures'

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
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

def replace_slices(img, gen, row_start, row_end, col_start, col_end):
    # Masks rows and columns to be replaced
    shape = tf.shape(img)
    rows = shape[1]
    cols = shape[2]
    i = tf.range(rows)
    row_mask = (row_start <= i) & (i < row_end)
    j = tf.range(cols)
    col_mask = (col_start <= j) & (j < col_end)
    # Full mask of replaced elements
    mask = row_mask[:, tf.newaxis] & col_mask
    # Add channel dimension to mask and cast
    mask = tf.cast(mask[:, :, tf.newaxis], img.dtype)
    # Compute result
    result = img * (1 - mask) + gen * mask
    return result

def patcher(tensors):
    img = tensors[1]
    gen = tensors[0]
    result = replace_slices(img, gen, 96, 160, 144, 240)
    return [result]

layer = Lambda(patcher,lambda x : [x[1]] )

outG = layer([x , a1])

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

def mask(batch):
    res = []
    x1 = 144    # w/2.0 - w/8.0
    x2 = 240    # w/2.0 + w/8.0  
    y1 = 96     # h/2.0 - h/8.0   
    y2 = 160    # h/2.0 + h/8.0   
        
    for im in batch:
        im2 = im.copy()
        cv2.rectangle(im2, (x1,y1) , ( x2,y2 ) ,(0,0,0),-1)
        res.append(np.array(im2))
    return res
        

gen_weights_path = './gan_weights/gen/weights.hdf5'
comb_weights_path = './gan_weights/comb/weights.hdf5'
disc_weights_path = './gan_weights/disc/weights.hdf5'


gen.load_weights(gen_weights_path)

source_path = './dataset/test'
save_path = './predictions/two'


datagen = ImageDataGenerator()
    
batches = datagen.flow_from_directory( 
        source_path ,target_size = (256,384),batch_size = 32, class_mode=None)
        
              
batch = batches.next()
batch /= 255.0
print('Shape::' + str(batch.shape))
batch = np.array(mask(batch))

print('Shape::' + str(batch.shape))

# Generate a batch of new images
gen_pred = gen.predict(batch)

i=1
for a in gen_pred:
    print(a.shape)
    a *= 255
    im = Image.fromarray(a.astype('uint8'))
    im.save(save_path + '/out' + str(i) + '.jpeg')
    i += 1





                                                                                
