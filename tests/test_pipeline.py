import sys
import os
import subprocess
import glob
import numpy as np

def parse_arff_data(filepath):
    lines = []
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if in_data:
                lines.append(line)
            elif line.lower() == '@data':
                in_data = True
    return sorted(lines)

def parse_csv_data(filepath):
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def compare_float_lines(lines1, lines2, tol=1e-3):
    vals1 = []
    for r in lines1:
        vals1.extend(r.split(','))

    vals2 = []
    for r in lines2:
        vals2.extend(r.split(','))

    assert len(vals1) == len(vals2), f"Element count mismatch: {len(vals1)} != {len(vals2)}"

    for i, (v1, v2) in enumerate(zip(vals1, vals2)):
        try:
            f1 = float(v1)
            f2 = float(v2)
            assert abs(f1 - f2) < tol, f"Value mismatch at index {i}: {v1} != {v2}"
        except ValueError:
            assert v1 == v2, f"Value mismatch at index {i}: {v1} != {v2}"

def test_pipeline():
    # 1. Execute run_facial_landmarks_prediction.py and verifies it runs correctly
    result1 = subprocess.run([sys.executable, os.path.abspath("run_facial_landmarks_prediction.py")], capture_output=True, text=True)
    assert result1.returncode == 0, f"run_facial_landmarks_prediction.py execution failed:\n{result1.stdout}\n{result1.stderr}"

    # 2. Execute main_facial_features_extraction.py and verifies it runs correctly
    result2 = subprocess.run([sys.executable, os.path.abspath('main_facial_features_extraction.py')], capture_output=True, text=True)
    assert result2.returncode == 0, f"main_facial_features_extraction.py execution failed:\n{result2.stdout}\n{result2.stderr}"

    # 3. Evaluate the files in folder test_results with files in test_ground_truth, their values should be the same within a small difference (tolerance). 
    test_results_dir = 'test_results'
    ground_truth_dir = 'test_ground_truth'

    # Check the ARFF files, parsing of the data is required before evaluating the values. 
    arff_result = os.path.join(test_results_dir, 'data_face-alignment.arff')
    arff_gt = os.path.join(ground_truth_dir, 'data_face-alignment_GT.arff')

    assert os.path.exists(arff_result), f"File {arff_result} not found in test_results"
    assert os.path.exists(arff_gt), f"File {arff_gt} not found in test_ground_truth"

    res_data = parse_arff_data(arff_result)
    gt_data = parse_arff_data(arff_gt)
    compare_float_lines(res_data, gt_data)

    # Check the VSC files, parsing of the data is required before evaluating the values.
    # face_aligment_dataset.csv verification
    csv_result = os.path.join(test_results_dir, 'face_aligment_dataset.csv')
    csv_gt = os.path.join(ground_truth_dir, 'face_aligment_dataset_GT.csv')

    assert os.path.exists(csv_result), f"File {csv_result} not found in test_results"
    assert os.path.exists(csv_gt), f"File {csv_gt} not found in test_ground_truth"

    res_data = sorted(parse_csv_data(csv_result))
    gt_data = sorted(parse_csv_data(csv_gt))
    compare_float_lines(res_data, gt_data)

    # smfeat_test_image_XXX.csv verification
    smfeat_files = glob.glob(os.path.join(test_results_dir, 'smfeat_test_image_*.csv'))

    assert len(smfeat_files) > 0, "No smfeat_test_image_*.csv files were generated."

    for smfeat_file in smfeat_files:
        filename = os.path.basename(smfeat_file)
        name, ext = os.path.splitext(filename)
        gt_filename = name + "_GT" + ext
        gt_file = os.path.join(ground_truth_dir, gt_filename)

        assert os.path.exists(gt_file), f"Ground truth file {gt_file} not found for {smfeat_file}"

        # For single files, do not sort the rows
        res_data = parse_csv_data(smfeat_file)
        gt_data = parse_csv_data(gt_file)
        compare_float_lines(res_data, gt_data)    

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
import main_facial_features_extraction as main

def test_gpFaceReg_synthetic():
    # Synthetic image (100x100 black square)
    image_face = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create 68 synthetic landmarks
    # Make them such that index 0 and 16 have different x and same y to test rotation
    data_keypts = np.zeros((FACIAL_LANDMARKS, 2), dtype=float)
    # Put jaw points at specific coordinates to cause a predictable rotation
    # delta_y = 0, delta_x = -60 -> no rotation in this test
    data_keypts[0] = [20, 20]
    data_keypts[16] = [80, 20]

    # Just fill the rest with something
    for i in range(1, 16):
        data_keypts[i] = [10, 10]
    for i in range(17, FACIAL_LANDMARKS):
        data_keypts[i] = [50, 50]

    rot_face, mat_landmarks = main.gpFaceReg(image_face, data_keypts)

    assert rot_face.shape == (100, 100, 3), "Rotated image shape mismatch"
    assert mat_landmarks.shape == (FACIAL_LANDMARKS, 2), "Rotated landmarks shape mismatch"
    assert mat_landmarks[0, 0] == data_keypts[0, 0], "Calculation error in rotation matrix"
    assert mat_landmarks[0, 1] == data_keypts[0, 1], "Calculation error in rotation matrix"
    assert abs(mat_landmarks[1, 0] - data_keypts[1, 0]) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[1, 1] - data_keypts[1, 1]) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[17, 0] - data_keypts[17, 0]) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[17, 1] - data_keypts[17, 1]) < MIN_EVAL, "Calculation error in rotation matrix"

    # Put jaw points at specific coordinates to cause a predictable rotation
    # delta_y = -5, delta_x = -60
    data_keypts[0] = [20, 20]
    data_keypts[16] = [80, 25]

    rot_face, mat_landmarks = main.gpFaceReg(image_face, data_keypts)

    assert rot_face.shape == (100, 100, 3), "Rotated image shape mismatch"
    assert mat_landmarks.shape == (FACIAL_LANDMARKS, 2), "Rotated landmarks shape mismatch"
    assert mat_landmarks[0, 0] == data_keypts[0, 0], "Calculation error in rotation matrix"
    assert mat_landmarks[0, 1] == data_keypts[0, 1], "Calculation error in rotation matrix"
    assert abs(mat_landmarks[1, 0] - 9.2041) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[1, 1] - 10.8650) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[17, 0] - 52.3877) < MIN_EVAL, "Calculation error in rotation matrix"
    assert abs(mat_landmarks[17, 1] - 47.4050) < MIN_EVAL, "Calculation error in rotation matrix"

def test_gpPtsExt_synthetic():
    data_keypts = np.zeros((FACIAL_LANDMARKS, 2), dtype=float)
    for i in range(FACIAL_LANDMARKS):
        data_keypts[i] = [i, i*2]

    coord_vector = main.gpPtsExt(data_keypts)

    # Verify landmarks are extracted correctly
    assert coord_vector.shape == (TOTAL_LANDMARK, 2), "Extracted points shape mismatch"
    # Eyebrow points 17-26 -> 10 points
    assert coord_vector[0, 0] == 17, "Extracted points shape misaligned"
    assert coord_vector[9, 0] == 26, "Extracted points shape misaligned"
    # Eye points 36-47 -> 12 points
    assert coord_vector[10, 0] == 36, "Extracted points shape misaligned"
    assert coord_vector[21, 0] == 47, "Extracted points shape misaligned"
    # Nose points 30-35 -> 6 points
    assert coord_vector[22, 0] == 30, "Extracted points shape misaligned"
    assert coord_vector[27, 0] == 35, "Extracted points shape misaligned"
    # Mouth points 48-67 -> 20 points
    assert coord_vector[28, 0] == 48, "Extracted points shape misaligned"
    assert coord_vector[47, 0] == 67, "Extracted points shape misaligned"
    # Jaw points 0, 16, 8 -> 3 points
    assert coord_vector[48, 0] == 0, "Extracted points shape misaligned"
    assert coord_vector[49, 0] == 16, "Extracted points shape misaligned"
    assert coord_vector[50, 0] == 8, "Extracted points shape misaligned"

def test_gpGetFMM_synthetic():
    norm_vector = np.zeros((TOTAL_LANDMARK, 2), dtype=float)
    # Provide simple coordinates
    for i in range(TOTAL_LANDMARK):
        norm_vector[i] = [i, i*2]

    # Prevent division by zero by separating some points used for distance
    norm_vector[28] = [0, 5]
    norm_vector[34] = [0, 4]
    norm_vector[37] = [0, 0]
    norm_vector[48] = [0, 0]
    norm_vector[49] = [10, 0]

    face_smr_data = np.zeros(TOTAL_SMR, dtype=np.float32)
    main.gpGetFMM(norm_vector, face_smr_data)

    # b_dist (5) > c_dist (4) -> face_smr_data[18] = b_dist / a_dist = 5 / 10 = 0.5
    assert abs(face_smr_data[18] - 0.5) < MIN_EVAL
    # sl_dist (22.36) > tl_dist (4.47) -> face_smr_data[19] = sl_dist / h_dist = 22.36 / 1 = 22.3607
    assert abs(face_smr_data[19] - 22.3607) < MIN_EVAL
    # su_dist (17.8885) > tu_dist (8.9443) -> face_smr_data[20] = su_dist / h_dist = 17.8885 / 1 = 17.8885
    assert abs(face_smr_data[20] - 17.8885) < MIN_EVAL
    # ml_dist (304.1769) > mr_dist (301.4662) -> face_smr_data[21] = ml_dist / h_dist = 304.1769 / 1 = 304.1769
    assert abs(face_smr_data[21] - 304.1769) < MIN_EVAL
    # d_dist (4.4721) > g_dist (15.6525) -> face_smr_data[25] = g_dist / a_dist = 15.6525 / 10 = 1.5652
    assert abs(face_smr_data[25] - 1.5652) < MIN_EVAL
    # e_dist (6.7082) > f_dist (13.4164) -> face_smr_data[26] = f_dist / a_dist = 13.4164 / 10 = 1.3416
    assert abs(face_smr_data[26] - 1.3416) < MIN_EVAL
    # i_dist / a_dist -> face_smr_data[28] = 13.4164 / 10 = 1.3416
    assert abs(face_smr_data[28] - 1.3416) < MIN_EVAL

def test_gpGetSMR_synthetic():
    norm_vector = np.zeros((TOTAL_LANDMARK, 2), dtype=float)
    # Provide simple coordinates
    for i in range(TOTAL_LANDMARK):
        norm_vector[i] = [i, i*2]

    face_smr_data = np.zeros(TOTAL_SMR, dtype=np.float32)

    # Avoid zero division
    norm_vector[0] = [0, 0]
    norm_vector[2] = [1, 1]
    norm_vector[4] = [10, 51]
    norm_vector[7] = [0, 20]
    norm_vector[9] = [0, 10]

    main.gpGetSMR(norm_vector, face_smr_data)

    # Check if angles are populated
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[0] - 90) < MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[1] -  86.9872) < MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[2] - 83.0470)< MIN_EVAL
    # a_dist > i_dist -> face_smr_data[3] = i_dist / a_dist = 12 / 13.6 = 0.8823
    assert abs(face_smr_data[3] - 0.8823)< MIN_EVAL
    # c_dist
    assert abs(face_smr_data[4] - 10e6)< MIN_EVAL
    # d_dist
    assert abs(face_smr_data[5] - 19)< MIN_EVAL
    # h_dist
    assert abs(face_smr_data[6] - 8.2)< MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[7] - 63.4349)< MIN_EVAL
    # gl_dist < gr_dist -> face_smr_data[8] = gl_dist / gr_dist = 6.7082 / 6.7082 = 1
    assert abs(face_smr_data[8] - 1)< MIN_EVAL
    # j_dist > k_dist -> face_smr_data[9] = k_dist / j_dist = 67.0820 / 84.9706 = 0.7895
    assert abs(face_smr_data[9] - 0.7895)< MIN_EVAL
    # m_dist > n_dist -> face_smr_data[10] = n_dist / m_dist = 17.8885 / 29.0689 = 0.6154
    assert abs(face_smr_data[10] - 0.6154)< MIN_EVAL
    # q_avg == r_avg -> face_smr_data[13] =  r_avg / q_avg = 6 / 6 = 1
    assert abs(face_smr_data[11] - 1) < MIN_EVAL
    # ql_dist > rr_dist -> face_smr_data[12] = rr_dist / ql_dist = 4.4721 / 8.9443 = 0.5
    assert abs(face_smr_data[12] - 0.5) < MIN_EVAL
    # qr_dist < rl_dist -> face_smr_data[13] =  qr_dist / rl_dist = 4.4721 / 8.9443 = 0.5
    assert abs(face_smr_data[13] - 0.5) < MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[14] - 63.4349)< MIN_EVAL
    # e_dist > f_dist -> face_smr_data[15] =  f_dist / e_dist = 40.2492 / 60.3738 = 0.6667
    assert abs(face_smr_data[15] - 0.6667) < MIN_EVAL
    # sl_dist > tl_dist -> face_smr_data[16] =  tl_dist / sl_dist = 4.4721 / 22.3607 = 0.2
    assert abs(face_smr_data[16] - 0.2) < MIN_EVAL
    # su_dist < tu_dist -> face_smr_data[17] =  tu_dist / su_dist = 8.9443 / 17.8885 = 0.5
    assert abs(face_smr_data[17] - 0.5) < MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[22] - 63.4349)< MIN_EVAL
    # abs(math.atan2(delta_y, delta_x) * 180 / math.pi)
    assert abs(face_smr_data[23] - 63.4349)< MIN_EVAL
    # o_dist > p_dist -> face_smr_data[24] =  p_dist / o_dist = 22.3607 / 31.3050 = 0.7143
    assert abs(face_smr_data[24] -0.7143) < MIN_EVAL
    # b_dist != 0 -> face_smr_data[27] =  l_dist / b_dist = 2.2361 / 17.8885 = 13
    assert abs(face_smr_data[27] - 13) < MIN_EVAL

