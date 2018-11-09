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
    return res

def mask(batch):
    res = []
    x1 = 144    # w/2.0 - w/8.0
    x2 = 240    # w/2.0 + w/8.0  
    y1 = 96     # h/2.0 - h/8.0   
    y2 = 160    # h/2.0 + h/8.0   
        
    for im in batch:
        im2 = im.copy()
          
        cv2.rectangle(im2, (x1,y1) , ( x2,y2 ) ,(255,255,255),-1)
        res.append(np.array(im2))
        
    return res

def myloss(y_true,y_pred):
    diff = y_pred - y_true
 
    zeros = np.zeros((256,384,3))
    zeros[96:160 , 144:240 , :]  = 1    
    diff = np.multiply(diff, zeros)
    
    return K.mean(K.square(diff), axis=-1)

    
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

     


#-------------------------------------
#train
#-------------------------------------
gen_weights_path = './gan_weights/gen/weights.hdf5'
comb_weights_path = './gan_weights/comb/weights.hdf5'
disc_weights_path = './gan_weights/disc/weights.hdf5'


#gen.load_weights(gen_weights_path)
#disc.load_weights(disc_weights_path)
#combined.load_weights(comb_weights_path)


datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
#        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
batches = datagen.flow_from_directory( 
        source_path ,target_size = (256,384),batch_size = 32, class_mode=None)
        

epochs = 500
sample_interval = 5


for epoch in range(epochs):
              
    batch = batches.next()
    batch /= 255.0
    batch_x = np.array(mask(batch))
    batch_y = np.array(batch)
            
    valid = np.ones((len(batch), 1))
    fake = np.zeros((len(batch), 1))
            
    # Generate a batch of new images
    gen_pred = gen.predict(batch_x)

    # Train the discriminator
    disc.trainable = True
    d_loss_real = disc.train_on_batch(batch_y, valid)
    d_loss_fake = disc.train_on_batch(gen_pred, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    #  Train Generator
    disc.trainable = False
    g_loss = combined.train_on_batch(batch_x, [batch_y, valid])

    # Plot the progress
    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        gen.save_weights(gen_weights_path)
        combined.save_weights(comb_weights_path)
        #disc.save_weights(disc_weights_path)












 









'''
val_datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
#        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

   
    
val_batches = val_datagen.flow_from_directory( 
        validation_path ,target_size = (256,384),batch_size = 32, class_mode=None)
    
'''

    
#----------------------------------------------------------
#predict
#----------------------------------------------------------

'''

def fixer(batches):
    for batch in batches:
        batch /= 255.0
        yield batch
        
def fixer2(img):
    img = np.array(img)
    img /= 255.0
    return img
        

model.load_weights(chkptPath)

test_path = './one'
save_path = './three'
predgen = ImageDataGenerator(rescale = 1./255)
test_images = predgen.flow_from_directory(test_path,class_mode=None,target_size = (256,384),
                                          batch_size=32)
arr = model.predict_generator(test_images)


i=1
for a in arr:
    print(a.shape)
    a *= 255
    im = Image.fromarray(a.astype('uint8'))
    im.save(save_path + '/out' + str(i) + '.jpeg')
    i += 1

'''


                                                                      
