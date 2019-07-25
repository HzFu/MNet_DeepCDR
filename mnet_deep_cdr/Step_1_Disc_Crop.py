# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path
from sys import modules

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

from mnet_deep_cdr import Model_DiscSeg as DiscModel
from mnet_deep_cdr.mnet_utils import BW_img, disc_crop, mk_dir, files_with_ext

disc_list = [400, 500, 600, 700, 800]
DiscROI_size = 800
DiscSeg_size = 640
CDRSeg_size = 400

data_type = '.jpg'
parent_dir = path.dirname(resource_filename(modules[__name__].__name__, '__init__.py'))
data_img_path = path.abspath(path.join(parent_dir, 'data', 'REFUGE-Training400', 'Training400', 'Glaucoma'))
label_img_path = path.abspath(path.join(parent_dir, 'data', 'Annotation-Training400',
                                        'Annotation-Training400', 'Disc_Cup_Masks', 'Glaucoma'))

data_save_path = mk_dir(path.join(parent_dir, 'training_crop', 'data'))
label_save_path = mk_dir(path.join(parent_dir, 'training_crop', 'label'))

file_test_list = files_with_ext(data_img_path, data_type)

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights(path.join(parent_dir, 'deep_model', 'Model_DiscSeg_ORIGA.h5'))

Disc_flat = None

for lineIdx, temp_txt in enumerate(file_test_list):
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))

    # load image
    org_img = np.asarray(image.load_img(path.join(data_img_path, temp_txt)))

    # load label
    org_label = np.asarray(image.load_img(path.join(label_img_path, temp_txt[:-4] + '.bmp')))[:, :, 0]
    new_label = np.zeros(np.shape(org_label) + (3,), dtype=np.uint8)
    new_label[org_label < 200, 0] = 255
    new_label[org_label < 100, 1] = 255

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])

    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)

    for disc_idx, DiscROI_size in enumerate(disc_list):
        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        label_region, _, _ = disc_crop(new_label, DiscROI_size, C_x, C_y)
        Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                           cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
        Label_flat = rotate(cv2.linearPolar(label_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                            cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)

        disc_result = Image.fromarray((Disc_flat * 255).astype(np.uint8))
        filename = '{}_{}.png'.format(temp_txt[:-4], DiscROI_size)
        disc_result.save(path.join(data_save_path, filename))
        label_result = Image.fromarray((Label_flat * 255).astype(np.uint8))
        label_result.save(path.join(label_save_path, filename))

plt.imshow(Disc_flat)
plt.show()
