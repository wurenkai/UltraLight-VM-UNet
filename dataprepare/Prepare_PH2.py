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

############################################################# Prepare PH2 data set #################################################
Dataset_add = './PH2/'
Tr_add = 'images'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2000 training samples
Data_train_2017    = np.zeros([200, height, width, channels])
Label_train_2017   = np.zeros([200, height, width])

print('Reading PH2')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2017[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'masks/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading PH2 finished')

################################################################ Make test sets ########################################    
# We consider 200 samples for testing

#Train_img      = Data_train_2017[0:1815,:,:,:]
#Validation_img = Data_train_2017[1815:1815+259,:,:,:]
Test_img       = Data_train_2017[0:200,:,:,:]

#Train_mask      = Label_train_2017[0:1815,:,:]
#Validation_mask = Label_train_2017[1815:1815+259,:,:]
Test_mask       = Label_train_2017[0:200,:,:]


#np.save('data_train', Train_img)
np.save('data_test' , Test_img)
#np.save('data_val'  , Validation_img)

#np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
#np.save('mask_val'  , Validation_mask)


