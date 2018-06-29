%
clear; close all;

img_folder = '../test_img/';
img_result = '../result/';
img_list = dir([img_folder '*.jpg']);

img_num = size(img_list, 1);

CDR_list = zeros([img_num, 1]);
for idx = 1:img_num
    img_name = img_list(idx).name;
    load([img_result img_name(1:end-4) '.mat']);
    CDR_list(idx) = fun_CalCDR( cat(3, ROI_map(:,:,1), ROI_map(:,:,2)) );
end