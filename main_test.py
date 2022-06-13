import tensorflow as tf
import config
import utils
import os
import matplotlib.pyplot as plt
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
import glob
import imageio as io
import numpy as np
import pandas as pd
import scipy.io as sio





# imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/",flag_multi_class = False,num_class = 3,image_prefix = "image",mask_prefix = "mask",image_as_gray = False,mask_as_gray = False)


root_dir = 'D:/Data/Deeplearning/Forest_fire/Andong_data'

dat_dir = root_dir + '/Sat_patch'
Label_dir =root_dir + '/Label'
ref_dataset = glob.glob(dat_dir+'/*.mat')	# the number of classes
ref_classes = os.listdir(dat_dir)
ref_classes.sort(key=len)
os.chdir(dat_dir)
Data_ref = np.zeros((len(ref_dataset),config.IMG_SHAPE,config.IMG_SHAPE,config.Band))

for i in range(len(ref_dataset)):
	file_name = ref_classes[i]
 	print(file_name)
 	temp = sio.loadmat(file_name)['test']
 	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,config.Band,)
 	Data_ref[i] = temp_data  # ch01

Label_dataset = glob.glob(Label_dir+'/*.mat')	# the number of classes
Label_classes = os.listdir(Label_dir)
Label_classes.sort(key=len)
os.chdir(Label_dir)
Data_label = np.zeros((len(Label_dataset),config.IMG_SHAPE,config.IMG_SHAPE,1))

for i in range(len(Label_dataset)):
 	file_name = Label_classes[i]
 	print(file_name)
 	temp = sio.loadmat(file_name)['label']
 	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,1)
 	Data_label[i] = temp_data  # ch01

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])