%%
% @author Gemma S. Parra-Dominguez gemmasparra at gmail.com
%
% If you use this code, please cite:
%      Parra-Dominguez, G.S.; Sanchez-Yanez, R.E.; Garcia-Capulin, C.H.
%      Facial Paralysis Detection on Images Using Key Point Analysis.
%      Appl. Sci. 2021, 11, 2435. https://doi.org/10.3390/app11052435
%%
path = 'test_results/';
vec_labels = [0; 1; 0; 0; 1];
cvs_name_dataset = 'face_aligment_dataset.csv';
arff_name_dataset = 'data_face-alignment.arff';
%%
gp_data_prepare_to_arff(path, vec_labels, cvs_name_dataset, arff_name_dataset)