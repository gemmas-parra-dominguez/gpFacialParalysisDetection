function [ data_output ] = gpNormalize( data_input, min_val, max_val )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
data_output = [];
[~,n] = size(data_input);
for ik = 1:n
    data = data_input(:,ik);    
    valmin = min(data);
    valmax = max(data);
    data_p = ((max_val-min_val)*(data-valmin)/(valmax-valmin)) + min_val;
    data_output = [data_output, data_p];
end

end

