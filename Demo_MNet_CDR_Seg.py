#
import numpy as np
import scipy.io as sio
import scipy.misc
from keras.preprocessing import image
from skimage.transform import rotate
from time import time
from utils import pro_process, BW_img

import cv2

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import Model_MNet as MNetModel

DiscROI_size = 800
CDRSeg_size = 400

pre_model_MNetSeg = 'Model_MNet_ORIGA_pretrain.h5'

data_type = '.jpg'
data_img_path = './test_img/'
data_save_path = 'result/'



if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
print(str(len(file_test_list)))

CDRSeg_model = MNetModel.DeepModel(size_set=CDRSeg_size)
CDRSeg_model.load_weights(pre_model_MNetSeg, by_name=True)

for lineIdx in range(0, len(file_test_list)):
    temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
    disc_region = np.asarray(image.load_img(data_img_path + temp_txt[0]))
    disc_region = scipy.misc.imresize(disc_region, (DiscROI_size, DiscROI_size, 3))

    run_time_start = time()
    Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size/2, DiscROI_size/2),
                                       DiscROI_size/2, cv2.WARP_FILL_OUTLIERS), -90)

    temp_img = pro_process(Disc_flat, CDRSeg_size)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [prob_6, prob_7, prob_8, prob_9, prob_10] = CDRSeg_model.predict(temp_img)

    run_time_end = time()

    prob_map = np.reshape(prob_10, (prob_10.shape[1], prob_10.shape[2], prob_10.shape[3]))
    disc_map = scipy.misc.imresize(prob_map[:, :, 0], (DiscROI_size, DiscROI_size))
    cup_map = scipy.misc.imresize(prob_map[:, :, 1], (DiscROI_size, DiscROI_size))

    disc_map[-round(DiscROI_size / 3):, :] = 0
    cup_map[-round(DiscROI_size / 2):, :] = 0
    disc_map = BW_img(disc_map, 0.5)
    cup_map = BW_img(cup_map, 0.5)

    De_disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                  DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    De_cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size/2, DiscROI_size/2),
                                 DiscROI_size/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    print('	Run time CDRSeg: ' + str(run_time_end - run_time_start) + '   Img number: ' + str(lineIdx + 1))

    sio.savemat(data_save_path + temp_txt[0][:-4] + '.mat', {'Disc_map': De_disc_map, 'Cup_map': De_cup_map})
    scipy.misc.imsave(data_save_path + temp_txt[0][:-4] + '_flat.png', Disc_flat)
    rgbArray = np.zeros((DiscROI_size, DiscROI_size, 3))
    rgbArray[..., 0] = BW_img(De_disc_map, 0.5)
    rgbArray[..., 1] = BW_img(De_cup_map, 0.5)
    scipy.misc.imsave(data_save_path + temp_txt[0][:-4] + '_seg.png', rgbArray)

