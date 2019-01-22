function [ CDR] = fun_CalCDR(disc_map, cup_map)
% 

[disc_x, ~] = find(disc_map>0);
[cup_x, ~] = find(cup_map>0);

if numel(disc_x)>0
    disc_dia = max(disc_x) - min(disc_x);
else
    disc_dia = 1;
end

if numel(cup_x)>0
    cup_dia = max(cup_x) - min(cup_x);
else
    cup_dia = 1;
end

CDR = cup_dia / disc_dia;

end

