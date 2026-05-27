import sys
import os
import subprocess
import glob

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
    # 1. Run the file command_line.py
    result1 = subprocess.run([sys.executable, os.path.abspath("command_line.py")], capture_output=True, text=True)
    assert result1.returncode == 0, f"command_line.py execution failed:\n{result1.stdout}\n{result1.stderr}"

    # 2. Run the file gpmain.py
    result2 = subprocess.run([sys.executable, os.path.abspath('gpmain.py')], capture_output=True, text=True)
    assert result2.returncode == 0, f"gpmain.py execution failed:\n{result2.stdout}\n{result2.stderr}"

    # 3. Compare the results file in folder test_results with files in test_ground_truth
    test_results_dir = 'test_results'
    ground_truth_dir = 'test_ground_truth'

    # Check data_face-alignment.arff
    arff_result = os.path.join(test_results_dir, 'data_face-alignment.arff')
    arff_gt = os.path.join(ground_truth_dir, 'data_face-alignment_GT.arff')

    assert os.path.exists(arff_result), f"File {arff_result} not found in test_results"
    assert os.path.exists(arff_gt), f"File {arff_gt} not found in test_ground_truth"

    res_data = parse_arff_data(arff_result)
    gt_data = parse_arff_data(arff_gt)
    compare_float_lines(res_data, gt_data)

    # Check face_aligment_dataset.csv
    csv_result = os.path.join(test_results_dir, 'face_aligment_dataset.csv')
    csv_gt = os.path.join(ground_truth_dir, 'face_aligment_dataset_GT.csv')

    assert os.path.exists(csv_result), f"File {csv_result} not found in test_results"
    assert os.path.exists(csv_gt), f"File {csv_gt} not found in test_ground_truth"

    res_data = sorted(parse_csv_data(csv_result))
    gt_data = sorted(parse_csv_data(csv_gt))
    compare_float_lines(res_data, gt_data)

    # Check all smfeat files (smfeat_test_image_XXX.csv)
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

    # 4. Do not include image comparison in the unit tests - explicitly omitting any .jpg checks.

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
import gpmain

def test_gpFaceReg_synthetic():
    # Synthetic image (100x100 black square)
    image_face = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create 68 synthetic landmarks
    # Make them such that index 0 and 16 have different x and y to test rotation
    data_keypts = np.zeros((68, 2), dtype=float)
    # Put jaw points at specific coordinates to cause a predictable rotation
    data_keypts[0] = [20, 20]
    data_keypts[16] = [80, 20] # delta_y = 0, delta_x = -60 -> angle 180

    # Just fill the rest with something
    for i in range(1, 16):
        data_keypts[i] = [20 + i*4, 20]
    for i in range(17, 68):
        data_keypts[i] = [50, 50]

    rot_face, mat_landmarks = gpmain.gpFaceReg(image_face, data_keypts)

    assert rot_face.shape == (100, 100, 3), "Rotated image shape mismatch"
    assert mat_landmarks.shape == (68, 2), "Rotated landmarks shape mismatch"

def test_gpPtsExt_synthetic():
    data_keypts = np.zeros((68, 2), dtype=float)
    for i in range(68):
        data_keypts[i] = [i, i*2]

    coord_vector = gpmain.gpPtsExt(data_keypts)

    assert coord_vector.shape == (51, 2), "Extracted points shape mismatch"
    # Eyebrow points 17-27 -> 10 points
    assert coord_vector[0, 0] == 17
    # Eye points 36-48 -> 12 points
    assert coord_vector[10, 0] == 36
    # Nose points 30-36 -> 6 points
    assert coord_vector[22, 0] == 30
    # Mouth points 48-68 -> 20 points
    assert coord_vector[28, 0] == 48
    # Jaw points 0, 16, 8 -> 3 points
    assert coord_vector[48, 0] == 0
    assert coord_vector[49, 0] == 16
    assert coord_vector[50, 0] == 8

def test_gpGetFMM_synthetic():
    norm_vector = np.zeros((51, 2), dtype=float)
    # Provide simple coordinates
    for i in range(51):
        norm_vector[i] = [i, i*2]

    # Prevent division by zero by separating some points used for distance
    norm_vector[48] = [0, 0]
    norm_vector[49] = [10, 0] # a_dist = 10

    norm_vector[37] = [0, 0]
    norm_vector[28] = [0, 5]
    norm_vector[34] = [0, 4]

    norm_vector[28] = [0, 5]  # b_dist = 5

    norm_vector[34] = [0, 4]  # c_dist = 4 (b_dist > c_dist)

    # set h_dist points
    # wait, h_dist uses 34 and 28, so we cannot overwrite them
    # b_dist uses 37 and 28. c_dist uses 37 and 34.
    norm_vector[37] = [0, 5] # b_dist to 28(0,0)=5
    norm_vector[28] = [0, 0]
    norm_vector[34] = [0, 1] # c_dist to 37(0,5)=4. h_dist to 28(0,0)=1


    face_smr_data = np.zeros(TOTAL_SMR, dtype=np.float32)
    gpmain.gpGetFMM(norm_vector, face_smr_data)

    # b_dist (5) > c_dist (4) -> face_smr_data[18] = b_dist / a_dist = 5 / 10 = 0.5
    assert abs(face_smr_data[18] - 0.5) < 1e-3

def test_gpGetSMR_synthetic():
    norm_vector = np.zeros((51, 2), dtype=float)
    # Provide simple coordinates
    for i in range(51):
        norm_vector[i] = [i, i*2]

    face_smr_data = np.zeros(TOTAL_SMR, dtype=np.float32)

    # Avoid zero division
    norm_vector[0] = [0, 0]
    norm_vector[9] = [10, 0] # delta_x = -10, delta_y = 0

    gpmain.gpGetSMR(norm_vector, face_smr_data)

    # Check if angles are populated
    assert face_smr_data[0] >= 0
