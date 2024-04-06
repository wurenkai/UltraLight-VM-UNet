# -*- coding: utf-8 -*-
##scipy==1.2.1

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256 # Enter the image size of the model.
width  = 256 # Enter the image size of the model.
channels = 3 # Number of image channels

train_number = 1000 # Randomly assign the number of images for generating the training set.
val_number = 200   # Randomly assign the number of images for generating the validation set.
test_number = 400  # Randomly assign the number of images for generating the test set.
all = int(train_number) + int(val_number) + int(test_number)

############################################################# Prepare your data set #################################################
Tr_list = glob.glob("images"+'/*.png')   # Images storage folder. The image type should be 24-bit png format.
# It contains 2594 training samples
Data_train_2018    = np.zeros([all, height, width, channels])
Label_train_2018   = np.zeros([all, height, width])

print('Reading')
print(len(Tr_list))
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]
    b = b[len(b)-8: len(b)-4]
    add = ("masks/" + b +'.png')  # Masks storage folder. The Mask type should be a black and white image of an 8-bit png (0 pixels for the background and 255 pixels for the target).
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2018[idx, :,:] = img2    
         
print('Reading your dataset finished')

################################################################ Make the training, validation and test sets ########################################    
Train_img      = Data_train_2018[0:train_number,:,:,:]  
Validation_img = Data_train_2018[train_number:train_number+val_number,:,:,:]
Test_img       = Data_train_2018[train_number+val_number:all,:,:,:]

Train_mask      = Label_train_2018[0:train_number,:,:]
Validation_mask = Label_train_2018[train_number:train_number+val_number,:,:]
Test_mask       = Label_train_2018[train_number+val_number:all,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)
