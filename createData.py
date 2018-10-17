from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K



def fixed_generator( batches ):
    for batch in batches:
        yield (batch,batch)
        

source_path = '/home/nirvan/Documents/image_restoration_project/dataset3'
target_path = '/home/nirvan/Documents/image_restoration_project/target'


datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.4,
        height_shift_range=0.4,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

   
i=0
for batch in datagen.flow_from_directory( 
        source_path, save_to_dir = target_path ,target_size = (400,1000),batch_size = 32, class_mode=None):
    if(i == 20):
        break
    i += 1
    
     



       
  
        
