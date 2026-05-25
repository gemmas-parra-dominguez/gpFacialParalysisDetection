import csv
import numpy as np
import math

def readCSV(file_name):
    """
    Reads a CSV file containing numeric data into a NumPy array.

    Args:
        file_name (str): The path to the CSV file to read.

    Returns:
        np.ndarray: A 2D NumPy array containing the parsed float values.
                    Returns an empty array if the file is not found.
    """
    try:
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                data.append([float(val) for val in row])
            return np.array(data, dtype=np.float32)
    except FileNotFoundError:
        return np.array([])

def writeCSV(filename, m):
    """
    Writes a 1D or 2D NumPy array (or any iterable of iterables) to a CSV file.

    Args:
        filename (str): The path to the output CSV file.
        m (np.ndarray or list): The data to write. If a 1D array is passed,
                                it is written as a single row.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if m.ndim == 1:
            writer.writerow(m)
        else:
            for row in m:
                writer.writerow(row)

def gpEuclideanDist(x_land, y_land, x_key, y_key):
    """
    Computes the Euclidean distance between two 2D points.

    Args:
        x_land (float): The x-coordinate of the first point.
        y_land (float): The y-coordinate of the first point.
        x_key (float): The x-coordinate of the second point.
        y_key (float): The y-coordinate of the second point.

    Returns:
        float: The straight-line Euclidean distance between the points.
    """
    return math.sqrt((x_land - x_key)**2 + (y_land - y_key)**2)
