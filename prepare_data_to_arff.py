import numpy as np
from utils import readCSV, writeCSV

def gpdata_extraction(path, smr_name, all_labels):
    all_data_set = []
    all_targets = []
    num_subj = len(all_labels)
    cont = 0

    for ik in range(num_subj):
        if ik < 10:
            name_file = f"{path}{smr_name}00{ik}.csv"
        elif ik < 100:
            name_file = f"{path}{smr_name}0{ik}.csv"
        else:
            name_file = f"{path}{smr_name}{ik}.csv"

        try:
            data_smr = readCSV(name_file)
            if data_smr.size > 0 and data_smr[0, 0] != -1:
                # Transpose the SMR column vector to a row vector
                if data_smr.ndim == 2 and data_smr.shape[1] == 1:
                    row_data = data_smr.T[0]
                elif data_smr.ndim == 2:
                    row_data = data_smr[0]
                else:
                    row_data = data_smr

                # Replace NaNs and Infs with 0 as in MATLAB
                row_data = np.nan_to_num(row_data, nan=0.0, posinf=0.0, neginf=0.0)

                all_data_set.append(row_data)
                all_targets.append(all_labels[cont][0])
        except Exception as e:
            pass
        cont += 1

    if len(all_data_set) == 0:
        return np.array([]), 0, 0

    all_data_set = np.array(all_data_set)
    all_targets = np.array(all_targets).reshape(-1, 1)

    data_set = np.hstack((all_data_set, all_targets))
    return data_set, data_set.shape[0], all_data_set.shape[1]

def gpValuesExtrac(data_set, num_features):
    mu_vals = np.mean(data_set, axis=0)
    std_vals = np.std(data_set, axis=0, ddof=1) # MATLAB uses ddof=1 by default
    feat_lims = np.zeros((num_features, 2))

    for ik in range(num_features):
        feat_lims[ik, 0] = -3 * std_vals[ik] + mu_vals[ik]
        feat_lims[ik, 1] = 3 * std_vals[ik] + mu_vals[ik]

    return feat_lims

def gpDataLim(data_set, feat_lims):
    data_out = data_set.copy()
    m, n = data_out.shape
    for ik in range(m):
        for jk in range(n):
            if data_out[ik, jk] < feat_lims[jk, 0]:
                data_out[ik, jk] = feat_lims[jk, 0]
            if data_out[ik, jk] > feat_lims[jk, 1]:
                data_out[ik, jk] = feat_lims[jk, 1]
    return data_out

def gpNormalize(data_input, min_val, max_val):
    data_output = np.zeros_like(data_input)
    n = data_input.shape[1]

    for ik in range(n):
        data = data_input[:, ik]
        valmin = np.min(data)
        valmax = np.max(data)
        if valmax - valmin != 0:
            data_p = ((max_val - min_val) * (data - valmin) / (valmax - valmin)) + min_val
        else:
            data_p = data # Avoid division by zero if all values are same
        data_output[:, ik] = data_p

    return data_output

def gp_data_prepare_to_arff(path, vec_labels, cvs_name_dataset, arff_name_dataset):
    smr_name = 'smfeat_test_image_'
    data_set, num_instances, num_features = gpdata_extraction(path, smr_name, vec_labels)

    if num_instances == 0:
        print("No valid instances found for ARFF creation.")
        return

    features_data = data_set[:, :num_features]
    labels_data = data_set[:, num_features].reshape(-1, 1)

    feat_lims = gpValuesExtrac(features_data, num_features)
    data_p = gpDataLim(features_data, feat_lims)

    fmatrix = gpNormalize(data_p, 0, 2)
    fmatrix = gpNormalize(fmatrix, -1, 1)

    data_set_norm = np.hstack((fmatrix, labels_data))

    # Random permutation
    np.random.seed(None) # Make sure it's random
    indices = np.random.permutation(num_instances)
    data_set_shuffled = data_set_norm[indices]

    # Save CSV
    writeCSV(path + cvs_name_dataset, data_set_shuffled)

    # Create ARFF
    with open(path + arff_name_dataset, 'w') as f:
        f.write('%\n')
        f.write('@relation face-alignment\n\n')

        for ik in range(num_features):
            f.write(f'@attribute  f{ik} REAL\n')

        f.write('@attribute  class {0,1}\n\n')
        f.write('@data\n')

        m, n = data_set_shuffled.shape
        f.write('%\n')
        f.write(f'% {m} instances\n')
        f.write('%\n')

        for ik in range(m):
            for jk in range(n - 1):
                f.write(f'{data_set_shuffled[ik, jk]:.6f},')
            f.write(f'{int(data_set_shuffled[ik, n-1])}\n')

        f.write('%\n')
        f.write('%')
