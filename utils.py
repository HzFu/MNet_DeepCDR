#

import numpy as np
from keras import backend as K
import scipy
from skimage.measure import label, regionprops

def pro_process(temp_img,input_size):
    img = np.asarray(temp_img).astype('float32')
    img = scipy.misc.imresize(img, (input_size, input_size, 3))
    return img

def BW_img(input, thresholding):
    binary = input > thresholding
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max+1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef2(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score = 0.5 * score0 + 0.5 * score1
    
    return score

def dice_coef_loss(y_true, y_pred):
    return -dice_coef2(y_true, y_pred)

