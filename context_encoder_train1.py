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
    res /= 255.0
    batch /= 255.0
    return (batch,res)


def fixed_generator( batches ):
    for batch in batches:
        yield cropper(batch)
        

source_path = './dataset/good_pictures'
validation_path = './dataset/validation'
#target_path = '/home/nirvan/Documents/image_restoration_project/target'


input_img = Input(shape=(256, 384, 3))

#new architecture 2
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


#new architecture
'''
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
x = Dense(256 * 16 * 16 ,activation='relu')(x)

x = Reshape((16,16,256))(x)
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


model = Model(input_img, x)
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='sgd')
#model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()

'''
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()
'''

chkptPath = './weights/weights.hdf5'
checkpointer = ModelCheckpoint(filepath=chkptPath, verbose=1, save_best_only=False,period=20,save_weights_only=True)

#-------------------------------------
#train
#-------------------------------------
model.load_weights(chkptPath)
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
    
#---show me de wei
'''
cnt = 0

for i in batches:
    if(cnt == 1):
        break
    cnt += 1
    #cv2.imwrite('/home/nirvan/Documents/image_restoration_project/temp/initial.jpg',i[0])
    
    
    _,ans = cropper(i)
    
    #for index,im in enumerate(ans):
    #    cv2.imwrite('/home/nirvan/Documents/image_restoration_project/temp/initial'+str(index)+'.jpg',im)
    for index,im in enumerate(ans):
        cv2.imwrite('/home/nirvan/Documents/image_restoration_project/temp2/cropped'+str(index)+'.jpg',im)
'''
#---show me de wei

model.fit_generator(
        fixed_generator(batches),steps_per_epoch=1,epochs=15000,callbacks=[checkpointer], validation_data=fixed_generator(val_batches),validation_steps = 1 )
        #samples_per_epoch=nb_train_samples,
        #nb_epoch=nb_epoch,
        #nb_val_samples=nb_validation_samples)

#model.save_weights('/home/nirvan/Documents/image_restoration_project/weights/weights2.h5')
        
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


       
  
        
