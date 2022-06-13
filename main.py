import tensorflow as tf
import config
import utils
import os
import matplotlib.pyplot as plt
import glob
import imageio as io
import numpy as np
import pandas as pd
import scipy.io as sio
from model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, plot_model
from IPython.display import clear_output
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
n_classes=3 # 분류할 class 개수
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU 실행 환경 확인
root_dir = 'D:/Data/Deeplearning/Forest_fire/02. Goseong/190408' # 데이터 저장되어 있은 경로

dat_dir = root_dir + '/Sat_patch'
Label_dir =root_dir + '/Label'
result_dir='D:/Data/Deeplearning/Forest_fire/02. Goseong/190408/model_result'

ref_dataset = glob.glob(dat_dir+'/*.mat')	# 해당 경로의 patch image 파일 불러오기
ref_classes = os.listdir(dat_dir)
ref_classes.sort(key=len) # 순서대로 정렬
os.chdir(dat_dir)
Data_ref = np.zeros((len(ref_dataset),config.IMG_SHAPE,config.IMG_SHAPE,config.Band)) # Patch image 저장할 np array 구성

###### Patch image 불러들이기 ######
for i in range(len(ref_dataset)):
	file_name = ref_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['train_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,config.Band,)
	Data_ref[i] = temp_data  # ch01

Label_dataset = glob.glob(Label_dir+'/*.mat') # 해당 경로의 label image 파일 불러오기
Label_classes = os.listdir(Label_dir)
Label_classes.sort(key=len) # 순서대로 정렬
os.chdir(Label_dir)
Data_label = np.zeros((len(Label_dataset),config.IMG_SHAPE,config.IMG_SHAPE,1)) # Label image 저장할 np array 구성

###### Labegl image 불러들이기 ######
for i in range(len(Label_dataset)):
	file_name = Label_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['label_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,1)
	Data_label[i] = temp_data  # ch01

Data_label=Data_label.astype(np.uint8) # Label image를 uint 8로 변경
np.unique(Data_label) # Label img 데이터 확인

labelencoder = LabelEncoder() # Multi class로 재구성하기 위한 변수 생성
k, n, h, w=Data_label.shape # Label image 개수 및 크기 불러오기
Data_label_reshaped=Data_label.reshape(-1,1) # Label image 재구성
Data_label_reshaped_encoded=labelencoder.fit_transform(Data_label_reshaped) # Label image 재구성
Data_label_reshaped_encoded_original_shape=Data_label_reshaped_encoded.reshape(k,n,h,w) # Label image 재구성
np.unique(Data_label_reshaped_encoded_original_shape) # 재구성된 label image 확인

Data_label_input=np.expand_dims(Data_label_reshaped_encoded_original_shape, axis=3) # 실제 학습을 위한 데이터 형식으로 재구성된 label 변환
data_train, data_test, label_train, label_test=train_test_split(Data_ref,Data_label_input,test_size = 0.30, random_state = 0) # train and test 이미지로 구분

#train_masks_cats = to_categorical(label_train, num_classes=n_classes)
#train_masks = train_masks_cats.reshape((label_train.shape[0], label_train.shape[1], label_train.shape[2], n_classes))

#test_masks_cat = to_categorical(label_test, num_classes=n_classes)
#test_masks = test_masks_cat.reshape((label_test.shape[0], label_test.shape[1], label_test.shape[2], n_classes))

full_masks_cat = to_categorical(Data_label_input, num_classes=n_classes)
full_masks = full_masks_cat.reshape((Data_label_input.shape[0], Data_label_input.shape[1], Data_label_input.shape[2], n_classes))

os.chdir(result_dir)
model = unet()
model_checkpoint = ModelCheckpoint('unet_Goseong.hdf5', monitor='loss',verbose=1, save_best_only=True)
history=model.fit(Data_ref, full_masks , batch_size=30, epochs=100, verbose=1, validation_split=0.3, shuffle=True, callbacks=[model_checkpoint])

########################### Test 진행 ###########################
###### Patch image 불러들이기 ######
testroot_dir='D:/Data/Deeplearning/Forest_fire/01. Andong/200429'
test_dir = testroot_dir + '/Test_patch'
testLabel_dir =testroot_dir + '/Test_label'

test_dataset = glob.glob(test_dir+'/*.mat')	# 해당 경로의 patch image 파일 불러오기
test_classes = os.listdir(test_dir)
test_classes.sort(key=len) # 순서대로 정렬
os.chdir(test_dir)
Test_data = np.zeros((len(test_dataset),config.IMG_SHAPE,config.IMG_SHAPE,config.Band)) # Patch image 저장할 np array 구성

for i in range(len(test_dataset)):
	file_name = test_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['train_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,config.Band,)
	Test_data[i] = temp_data  # ch01

testlabel_dataset = glob.glob(testLabel_dir+'/*.mat') # 해당 경로의 label image 파일 불러오기
testlabel_classes = os.listdir(testLabel_dir)
testlabel_classes.sort(key=len) # 순서대로 정렬
os.chdir(testLabel_dir)
test_label = np.zeros((len(testlabel_dataset),config.IMG_SHAPE,config.IMG_SHAPE,1)) # Label image 저장할 np array 구성

###### Labegl image 불러들이기 ######
for i in range(len(testlabel_dataset)):
	file_name = testlabel_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['label_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,1)
	test_label[i] = temp_data  # ch01

test_label=test_label.astype(np.uint8) # Label image를 uint 8로 변경
np.unique(test_label) # Label img 데이터 확인

k, n, h, w=test_label.shape # Label image 개수 및 크기 불러오기
test_label_reshaped=test_label.reshape(-1,1) # Label image 재구성
test_label_reshaped_encoded=labelencoder.fit_transform(test_label_reshaped) # Label image 재구성
test_label_reshaped_encoded_original_shape=test_label_reshaped_encoded.reshape(k,n,h,w) # Label image 재구성
np.unique(test_label_reshaped_encoded_original_shape) # 재구성된 label image 확인

test_label_input=np.expand_dims(test_label_reshaped_encoded_original_shape, axis=3) # 실제 학습을 위한 데이터 형식으로 재구성된 label 변환
test_masks_cat2 = to_categorical(test_label_input, num_classes=n_classes)
test_masks2 = test_masks_cat2.reshape((test_label_input.shape[0], test_label_input.shape[1], test_label_input.shape[2], n_classes))


_, acc = model.evaluate(Test_data, test_masks2)
print("Accuracy is = ", (acc * 100.0), "%")

###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, loss, 'yo', label='Training loss')
plt.plot(epochs, val_loss, 'pg', label='Validation loss')
plt.plot(epochs, acc, 'k', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy & Loss of training and validation ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###
# import random
# test_img_number = random.randint(0, len(data_test))
# test_img = data_test[test_img_number]
# ground_truth=test_masks[test_img_number]
# prediction = (model.predict(data_test))
# sample_number=0
# predicted_img=np.argmax(prediction, axis=3)[sample_number,:,:]
#
#
# plt.figure(figsize=(12, 8))
# plt.subplot(131)
# plt.title('Testing Image')
# plt.imshow(data_test[sample_number,:,:,0:3], cmap='jet')
# plt.subplot(132)
# plt.title('Testing Label')
# plt.imshow(test_masks[sample_number,:,:,:], cmap='jet')
# plt.subplot(133)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()

#####################################################################

# Predict on large image

# Apply a trained model on large image

from patchify import patchify, unpatchify
from sklearn.feature_extraction import image
import spectral
large_image = sio.loadmat('D:/Data/Deeplearning/Forest_fire/01. Andong/200429/Andong200429_pad.mat')['large_img']
label_image = sio.loadmat('D:/Data/Deeplearning/Forest_fire/01. Andong/200429/Andong200429_label.mat')['large_pad']
lar_row,lar_col,lar_band=large_image.shape
ground_truth = spectral.imshow(classes = label_image,figsize =(7,7))
# This will split the image into small images of shape [3,3]
patches = patchify(large_image, (32, 32, 11), step=32)  #Step=256 for 256 patches means no overlap
patches.shape
predicted_patches = []
for i in range(patches.shape[0]):
	for j in range(patches.shape[1]):


		single_patch = patches[i, j, :, :]
		single_patch_prediction = (model.predict(single_patch))
		single_patch_predicted_img = np.argmax(single_patch_prediction, axis=3)[0, :, :]

		predicted_patches.append(single_patch_predicted_img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 1, 32, 32,1))
predicted_patches_reshaped.shape
reconstructed_image = unpatchify(predicted_patches_reshaped, (lar_row, lar_col, 1))
reconstructed_image_re=reconstructed_image[0:886,0:976]
k_label=label_image[0:886,0:976]
spectral.imshow(classes = reconstructed_image_re,figsize =(7,7))
spectral.imshow(classes = k_label,figsize =(7,7))

# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)

#####################################################################

#To calculate I0U for each class...
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
os.chdir(result_dir)
model.load_weights('unet_Goseong.hdf5')
y_pred=model.predict(Test_data)
y_pred_argmax=np.argmax(y_pred, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_label [:,:,:,0], y_pred_argmax)
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] +values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
label_image_tt=np.expand_dims(k_label,axis=2)
tttt=reconstructed_image_re
kappa = cohen_kappa_score(label_image_tt.flatten(), tttt.flatten())
f1_sc=f1_score(label_image_tt.flatten(), tttt.flatten(), average='micro')

classification = classification_report(label_image_tt.flatten(), tttt.flatten())
print(classification)
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("Mean IoU =", IOU_keras.result().numpy())
print("Kappa =", kappa )
print("F1-score =", f1_sc )
import cv2
cv2.imwrite('D:/Data/Deeplearning/Forest_fire/02. Goseong/190408/Goseong_Andong_result.tif',reconstructed_image_re)