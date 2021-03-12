function [ data_out ] = gpDataLim( data_set, feat_lims )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
%%
[m,n] = size(data_set);
data_out = data_set;
for ik=1:m
    for jk=1:n
        if(data_out(ik,jk) < feat_lims(jk,1))
            data_out(ik,jk) = feat_lims(jk,1);
        end
        if(data_out(ik,jk) > feat_lims(jk,2))
            data_out(ik,jk) = feat_lims(jk,2);
        end
    end
end

end

