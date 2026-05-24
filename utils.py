import csv
import numpy as np
import math

def readCSV(file_name):
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
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if m.ndim == 1:
            writer.writerow(m)
        else:
            for row in m:
                writer.writerow(row)

def gpEuclideanDist(x_land, y_land, x_key, y_key):
    return math.sqrt((x_land - x_key)**2 + (y_land - y_key)**2)
