function [data_set] = gpdata_extraction(path, smr_name, all_labels)
% gpdata_extraction this function reads information from a csv file and
%       store the information with the corresponding label.
%   path:           where the data is
%   smr_name:       default file name
%   all_labels:     labels for all instances (samples) stored in path
%   data_set:       a matrix containing all the information
%% Data extraction
all_data_set = [];
all_targets = [];
num_subj = length(all_labels);
%%
cont = 1;
for ik = 0:num_subj-1
    if (ik < 10)
        name_file = strcat(path,smr_name,'00',num2str(ik),'.csv');
    elseif(ik < 100)
        name_file = strcat(path,smr_name,'0',num2str(ik),'.csv');
    else
        name_file = strcat(path,smr_name,num2str(ik),'.csv');
    end
    data_smr = csvread(name_file);
    if ((data_smr(1,1) ~= -1))
        all_data_set = [all_data_set; data_smr'];
        all_targets = [all_targets; all_labels(cont,1)];
    end
    cont = cont + 1;
end
%% Data prep
[m,n] = size(all_data_set);
for ik = 1:m
    for jk = 1:n
        if (isnan(all_data_set(ik,jk)) || isinf(all_data_set(ik,jk)))
            all_data_set(ik,jk) = 0;
        end
    end
end
data_set = [all_data_set, all_targets];
end