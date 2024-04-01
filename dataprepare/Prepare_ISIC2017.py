# -*- coding: utf-8 -*-
"""
Code created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""

"""
Reminder added on December 6, 2023. 
Reminder Created on Wed Dec 6 2023
@author: Renkai Wu
1.Note that the scipy package should need to be degraded. Otherwise, you need to modify the following code. ##scipy==1.2.1
2.Add a name that displays the file to be processed. If it does not appear, the output npy file is incorrect.
"""

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = './ISIC2017/'
Tr_add = 'ISIC2017_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2000 training samples
Data_train_2017    = np.zeros([2000, height, width, channels])
Label_train_2017   = np.zeros([2000, height, width])

print('Reading ISIC 2017')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2017[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC2017_Task1_Training_GroundTruth/' + b +'_segmentation.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1250 samples for training, 150 samples for validation and 600 samples for testing

Train_img      = Data_train_2017[0:1250,:,:,:]
Validation_img = Data_train_2017[1250:1250+150,:,:,:]
Test_img       = Data_train_2017[1250+150:2000,:,:,:]

Train_mask      = Label_train_2017[0:1250,:,:]
Validation_mask = Label_train_2017[1250:1250+150,:,:]
Test_mask       = Label_train_2017[1250+150:2000,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)


