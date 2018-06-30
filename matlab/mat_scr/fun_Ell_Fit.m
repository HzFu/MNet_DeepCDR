function [ Ellp ] = fun_Ell_Fit( prob_map,img_h,img_w, IsFit)
%FUN_COMBINE_MAP Summary of this function goes here
%   Detailed explanation goes here

Ellp.raw_map = imresize(prob_map > 0, [img_h,img_w], 'nearest');

if IsFit == 1
    Ellp.ellp_para = fun_DiscFit( Ellp.raw_map );
    Ellp.fit_map = maskEllipse(img_h,img_w, Ellp.ellp_para.z(2), Ellp.ellp_para.z(1),...
        Ellp.ellp_para.a, Ellp.ellp_para.b, Ellp.ellp_para.alpha);
end


end