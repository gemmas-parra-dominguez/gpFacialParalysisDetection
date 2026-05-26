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
