import pytest
import numpy as np
import os
import sys
from unittest.mock import patch, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import readCSV, writeCSV, gpEuclideanDist

def test_readCSV_success(tmp_path):
    """
    Test that readCSV correctly reads a valid CSV file containing numeric data
    and returns a 2D numpy array of floats.
    """
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("1.0,2.0\n3.5,4.5\n")

    result = readCSV(str(p))
    expected = np.array([[1.0, 2.0], [3.5, 4.5]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)

def test_readCSV_file_not_found():
    """
    Test that readCSV returns an empty numpy array when the specified file
    is not found.
    """
    result = readCSV("non_existent_file.csv")
    assert result.size == 0
    assert isinstance(result, np.ndarray)

def test_writeCSV_1d_array(tmp_path):
    """
    Test that writeCSV correctly writes a 1D numpy array as a single row
    in a CSV file.
    """
    p = tmp_path / "output_1d.csv"
    data = np.array([1.5, 2.5, 3.5])

    writeCSV(str(p), data)

    content = p.read_text()
    assert content.strip() == "1.5,2.5,3.5"

def test_writeCSV_2d_array(tmp_path):
    """
    Test that writeCSV correctly writes a 2D numpy array as multiple rows
    in a CSV file.
    """
    p = tmp_path / "output_2d.csv"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    writeCSV(str(p), data)

    content = p.read_text()
    # Handle potentially different line endings on Windows vs Linux
    lines = content.strip().replace('\r\n', '\n').split('\n')
    assert len(lines) == 2
    assert lines[0] == "1.0,2.0"
    assert lines[1] == "3.0,4.0"

def test_gpEuclideanDist():
    """
    Test that gpEuclideanDist computes the correct Euclidean distance
    between two 2D points.
    """
    p1 = [0.0, 0.0]
    p2 = [3.0, 4.0]
    dist = gpEuclideanDist(p1, p2)
    assert dist == 5.0
