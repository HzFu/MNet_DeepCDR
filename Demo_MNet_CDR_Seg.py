#
import numpy as np
import scipy.io as sio
import scipy.misc
from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from time import time
from utils import pro_process, BW_img
from PIL import Image
from matplotlib.pyplot import imshow

import cv2
import os

import Model_DiscSeg as DiscModel
import Model_MNet as MNetModel

DiscROI_size = 800
DiscSeg_size = 640
CDRSeg_size = 400

data_type = '.jpg'
data_img_path = './test_img/'
data_save_path = 'result/'



if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
print(str(len(file_test_list)))

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('Model_DiscSeg_ORIGA_pretrain.h5')

CDRSeg_model = MNetModel.DeepModel(size_set=CDRSeg_size)
CDRSeg_model.load_weights('Model_MNet_ORIGA_pretrain.h5')


for lineIdx in range(0, len(file_test_list)):
    temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]

    # load image
    org_img = np.asarray(image.load_img(data_img_path + temp_txt[0]))

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3))*255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [prob_6, prob_7, prob_8, prob_9, prob_10] = DiscSeg_model.predict([temp_img])
    disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
    regions = regionprops(label(disc_map))
    C_x = regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size
    C_y = regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size
    disc_region = org_img[int(C_x-DiscROI_size/2):int(C_x+DiscROI_size/2),
                  int(C_y-DiscROI_size/2):int(C_y+DiscROI_size/2), ]

    # Disc and Cup segmentation by M-Net
    run_time_start = time()
    Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size/2, DiscROI_size/2), 
                DiscROI_size/2, cv2.WARP_FILL_OUTLIERS), -90)

    temp_img = pro_process(Disc_flat, CDRSeg_size)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [prob_6, prob_7, prob_8, prob_9, prob_10] = CDRSeg_model.predict(temp_img)
    run_time_end = time()

    # Extract mask
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

    print(' Run time MNet: ' + str(run_time_end - run_time_start) + '   Img number: ' + str(lineIdx + 1))

    # Save mask
    ROI_result = np.array(BW_img(De_disc_map, 0.5), dtype=int) + np.array(BW_img(De_cup_map, 0.5), dtype=int)

    Img_result = np.zeros((org_img.shape[0],org_img.shape[1]), dtype=int)
    Img_result[int(C_x-DiscROI_size/2):int(C_x+DiscROI_size/2), 
               int(C_y-DiscROI_size/2):int(C_y+DiscROI_size/2)] = ROI_result

    sio.savemat(data_save_path + temp_txt[0][:-4] + '.mat', {'Img_map': Img_result, 'ROI_map': ROI_result})


