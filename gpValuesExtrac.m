function [ feat_lims, mu_vals, std_vals ] = gpValuesExtrac(data_set, num_features)
%gpValuesExtrac this function computes the mean and standard deviation of
%       the input data and computes features value limits (up and low).
%   feat_lims:  a vector containing the feature limits
%   mu_vals:    mean values per feature
%   std_vals:   standard deviation values per feature
%%
mu_vals = mean(data_set);
std_vals = std(data_set);
feat_lims = zeros(num_features,2);
for ik = 1:num_features
    feat_lims(ik,1) = -3.*std_vals(1,ik) + mu_vals(1,ik);
    feat_lims(ik,2) = 3.*std_vals(1,ik) + mu_vals(1,ik);
end
end