import os,numpy as np,cv2
#from PIL import Image

path = '/home/nirvan/Documents/image_restoration_project/dataset/good_pictures/lol/'

for f in os.listdir(path):
    img = np.array(cv2.imread(path+'/'+f))
    nn = np.isnan(img)
    if( False in nn ):
        print(f+' has nan')

width=0
ratio = 0

n = len(os.listdir(path))

for f in os.listdir(path):
    img = Image.open(path+'/'+f)
    w , h = img.size
    
    ratio += w/h
    
    width += w
    
    
print('Average width = ' + str(width/n) )    
print('Average w/h = ' + str(ratio/n) )
    
    
    