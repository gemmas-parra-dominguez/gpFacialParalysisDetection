function [ data_set, num_instances, num_features ] = gpdata_read(path, all_labels)
% gpdata_read this function reads the information and put it into a mtrix
%             with the corresponding labels in the last column.
%   path:           where the data is
%   all_labels:     labels for all instances (samples) stored in path
%   data_set:       a matrix containing all the information
%   num_instances:  number of instances (samples) in the path
%   num_features:   number of features, 32 by default
%% default file names
smr_name = 'smfeat_test_image_';
%% data extraction
data_set = gpdata_extraction(path, smr_name, all_labels);
[m,n] = size(data_set);
num_features = n-1;
num_instances = m;
end