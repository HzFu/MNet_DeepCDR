function [ ellp ] = fun_DiscFit( tmp_map )
%FUN_FITELLIP Summary of this function goes here
%   Detailed explanation goes here

edge_disc = edge(tmp_map);

[tmp_y,tmp_x,~] = find(uint8(edge_disc)>0);

[ellp.z, ellp.a, ellp.b, ellp.alpha] = fitellipse([tmp_x, tmp_y], 'linear');


npts = 100;
t = linspace(0, 2*pi, npts);

% Rotation matrix
Q = [cos(ellp.alpha), -sin(ellp.alpha); sin(ellp.alpha) cos(ellp.alpha)];
% Ellipse points
ellp.X =  Q * [ellp.a * cos(t); ellp.b * sin(t)] + repmat(ellp.z, 1, npts);


end

