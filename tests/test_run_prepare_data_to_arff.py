import pytest
import numpy as np
import os
import sys
from unittest.mock import patch, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_prepare_data_to_arff import gpdata_extraction, gpValuesExtrac, gpDataLim, gpNormalize, gp_data_prepare_to_arff
from utils import writeCSV

def test_gpdata_extraction(tmp_path):
    """
    Test gpdata_extraction properly reads SMR features from multiple CSV files,
    appends labels, and handles missing files gracefully.
    """
    d = tmp_path / "data"
    d.mkdir()

    # Create two dummy CSV files matching the naming scheme.
    # labels: [1], [0], [1]
    # files: smfeat_test_image_000.csv, smfeat_test_image_001.csv

    smr0 = d / "smfeat_test_image_000.csv"
    writeCSV(str(smr0), np.array([[1.0], [2.0], [3.0]])) # Transposed later

    smr1 = d / "smfeat_test_image_001.csv"
    writeCSV(str(smr1), np.array([[4.0], [5.0], [6.0]]))

    # Intentionally omit smfeat_test_image_002.csv to test missing file logic.

    labels = [[1], [0], [1]]
    data_set, num_instances, num_features = gpdata_extraction(str(d) + "/", "smfeat_test_image_", labels)

    # Should only process the two existing files.
    assert num_instances == 2
    assert num_features == 3

    expected_data = np.array([
        [1.0, 2.0, 3.0, 1.0],
        [4.0, 5.0, 6.0, 0.0]
    ])

    np.testing.assert_array_equal(data_set, expected_data)

def test_gpValuesExtrac():
    """
    Test gpValuesExtrac calculates the correct -3 and +3 standard deviation
    limits for each feature in a dataset.
    """
    # Create a small dataset: 3 instances, 2 features
    # Feature 0: 1, 2, 3 -> mean 2, std 1
    # Feature 1: 4, 4, 4 -> mean 4, std 0
    data = np.array([
        [1.0, 4.0],
        [2.0, 4.0],
        [3.0, 4.0]
    ])

    limits = gpValuesExtrac(data, 2)

    # std of [1, 2, 3] with ddof=1 is 1.0. Limits: 2 - 3(1) = -1, 2 + 3(1) = 5
    # std of [4, 4, 4] with ddof=1 is 0.0. Limits: 4 - 3(0) = 4, 4 + 3(0) = 4
    expected = np.array([
        [-1.0, 5.0],
        [4.0, 4.0]
    ])

    np.testing.assert_array_equal(limits, expected)

def test_gpDataLim():
    """
    Test gpDataLim correctly clips dataset values according to provided
    lower and upper limits.
    """
    data = np.array([
        [-2.0, 10.0],
        [3.0, 2.0],
        [6.0, -1.0]
    ])

    # Limits: F0 [-1, 5], F1 [0, 5]
    limits = np.array([
        [-1.0, 5.0],
        [0.0, 5.0]
    ])

    clipped = gpDataLim(data, limits)

    expected = np.array([
        [-1.0, 5.0],
        [3.0, 2.0],
        [5.0, 0.0]
    ])

    np.testing.assert_array_equal(clipped, expected)

def test_gpNormalize():
    """
    Test gpNormalize correctly scales dataset values to the specified
    [min_val, max_val] range, handling features with zero variance.
    """
    data = np.array([
        [0.0, 5.0],
        [5.0, 5.0],
        [10.0, 5.0]
    ])

    # Normalize to [-1, 1]
    norm = gpNormalize(data, -1, 1)

    # Feature 0: min 0, max 10. Scaling [0, 5, 10] to [-1, 1] gives [-1, 0, 1]
    # Feature 1: min 5, max 5. No variance, remains [5, 5, 5]
    expected = np.array([
        [-1.0, 5.0],
        [0.0, 5.0],
        [1.0, 5.0]
    ])

    np.testing.assert_array_equal(norm, expected)

def test_gp_data_prepare_to_arff(tmp_path):
    """
    Test gp_data_prepare_to_arff correctly processes data, writes a normalized
    CSV file, and generates a valid ARFF file.
    """
    d = tmp_path / "data"
    d.mkdir()

    # Create two dummy CSV files matching the naming scheme.
    smr0 = d / "smfeat_test_image_000.csv"
    writeCSV(str(smr0), np.array([[1.0], [2.0], [3.0]]))

    smr1 = d / "smfeat_test_image_001.csv"
    writeCSV(str(smr1), np.array([[4.0], [5.0], [6.0]]))

    labels = [[1], [0]]
    csv_name = "test_dataset.csv"
    arff_name = "test_dataset.arff"

    gp_data_prepare_to_arff(str(d) + "/", labels, csv_name, arff_name)

    # Verify CSV was created
    csv_path = d / csv_name
    assert csv_path.exists()

    # Read the CSV to check its contents
    # Since we shuffle, the order might change, but the data must be normalized.
    from utils import readCSV
    csv_data = readCSV(str(csv_path))
    assert csv_data.shape == (2, 4) # 2 instances, 3 features + 1 label

    # Verify ARFF was created
    arff_path = d / arff_name
    assert arff_path.exists()

    arff_content = arff_path.read_text()
    assert "@relation face-alignment" in arff_content
    assert "@attribute  f0 REAL" in arff_content
    assert "@attribute  class {0,1}" in arff_content
    assert "@data" in arff_content
