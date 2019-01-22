#
import numpy as np
import scipy.io as sio
from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from mnet_utils import BW_img, disc_crop, mk_dir, return_list
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import Model_DiscSeg as DiscModel

disc_list = [400, 500, 600, 700, 800]
DiscROI_size = 800
DiscSeg_size = 640
CDRSeg_size = 400

data_type = '.jpg'
data_img_path = '../data/REFUGE-Training400/Training400/Glaucoma/'
label_img_path = '../data/Annotation-Training400/Annotation-Training400/Disc_Cup_Masks/Glaucoma/'

data_save_path = mk_dir('../training_crop/data/')
label_save_path = mk_dir('../training_crop/label/')


file_test_list = return_list(data_img_path, data_type)

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('./deep_model/Model_DiscSeg_ORIGA.h5')

for lineIdx in range(len(file_test_list)):

    temp_txt = file_test_list[lineIdx]
    print(' Processing Img ' + str(lineIdx + 1) + ': ' + temp_txt)

    # load image
    org_img = np.asarray(image.load_img(data_img_path + temp_txt))

    # load label
    org_label = np.asarray(image.load_img(label_img_path + temp_txt[:-4] + '.bmp'))[:,:,0]
    new_label = np.zeros(np.shape(org_label) + (3,), dtype=np.uint8)
    new_label[org_label < 200, 0] = 255
    new_label[org_label < 100, 1] = 255

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3))*255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])

    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)

    for disc_idx in range(len(disc_list)):
        DiscROI_size = disc_list[disc_idx]
        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        label_region, _, _ = disc_crop(new_label, DiscROI_size, C_x, C_y)
        Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size/2, DiscROI_size/2), DiscROI_size/2,
                                           cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS), -90)
        Label_flat = rotate(cv2.linearPolar(label_region, (DiscROI_size/2, DiscROI_size/2), DiscROI_size/2,
                                           cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS), -90)

        disc_result = Image.fromarray((Disc_flat * 255).astype(np.uint8))
        disc_result.save(data_save_path + temp_txt[:-4] + '_' + str(DiscROI_size) + '.png')
        label_result = Image.fromarray((Label_flat * 255).astype(np.uint8))
        label_result.save(label_save_path + temp_txt[:-4] + '_' + str(DiscROI_size) + '.png')

plt.imshow(Disc_flat)
plt.show()


