%
clear; close all;
addpath(genpath('./PD_toolbox/'));
addpath('./mat_scr/');

img_folder = '../test_img/';
img_result = '../result/';
img_list = dir([img_folder '*.jpg']);

img_num = size(img_list, 1);

CDR_list = zeros([img_num, 1]);
for idx = 1:img_num
    img_name = img_list(idx).name;
    load([img_result img_name(1:end-4) '.mat']);

    [img_h, img_w, img_c] = size(Img_map);
    Disc_map = fun_Ell_Fit( Img_map>0, img_h, img_w, 1);
    Cup_map = fun_Ell_Fit( Img_map>1, img_h, img_w, 1);
    CDR_list(idx) = fun_CalCDR( Disc_map.fit_map, Cup_map.fit_map);
end