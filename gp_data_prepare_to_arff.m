function [  ] = gp_data_prepare_to_arff( path, vec_labels, cvs_name_dataset, arff_name_dataset )
%gp_data_prepare_to_arff this is the main function to call to create the
%           ARFF file from the extracted features for evaluating on WEKA.
%   path:       where the data is, csv files should be named as "smfeat_test_image_xxx.csv"
%   vec_labels: labels for all instances (samples) stored in path, where:
%               0 goes for healthy
%               1 goes for unhealthy
%% data extraction from csv files
[ data_set, num_instances, num_features ] = gpdata_read(path, vec_labels);
%% data preparation
label = num_features + 1;
feat_lims = gpValuesExtrac(data_set(:,1:num_features),num_features);
data_p = gpDataLim(data_set(:,1:num_features),feat_lims);
fmatrix = gpNormalize(data_p, 0, 2);
fmatrix = gpNormalize(fmatrix, -1, 1);
data_set = [fmatrix,data_set(:,label)];
ik = randperm(num_instances);
data_set = data_set(ik,:);
csvwrite(strcat(path,cvs_name_dataset),data_set);
%% ARFF creation
fileID = fopen(strcat(path,arff_name_dataset),'w');
fprintf(fileID,strcat('%%','\n'));
fprintf(fileID,strcat('@relation face-alignment','\n'));
fprintf(fileID,strcat('\n'));
for ik = 1:num_features
    fprintf(fileID,strcat('@attribute ',' f',num2str(ik-1),' REAL','\n'));
end
fprintf(fileID,strcat('@attribute ',' class',' {0,1}','\n'));
fprintf(fileID,strcat('\n'));
fprintf(fileID,strcat('@data','\n'));
[m,n] = size(data_set);
fprintf(fileID,strcat('%%','\n'));
fprintf(fileID,strcat('%% ',num2str(m),' instances','\n'));
fprintf(fileID,strcat('%%','\n'));
for ik = 1:m
    for jk = 1:n-1
        fprintf(fileID,strcat(num2str(data_set(ik,jk)),','));
    end
    fprintf(fileID,strcat(num2str(data_set(ik,n)),'\n'));
end
fprintf(fileID,strcat('%%','\n'));
fprintf(fileID,strcat('%%'));
fclose(fileID);
end