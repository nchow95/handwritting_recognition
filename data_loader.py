import gzip
import numpy as np

def load_training_images():
    file_data = gzip.open('data/training_data.gz', 'rb')
    data = np.array(list(file_data.read()))
    return np.divide(data[16:], 255.0)

def load_training_labels():
    file_data = gzip.open('data/training_labels.gz', 'rb')
    labels = file_data.read()
    mat = []
    for value in labels[8:]:
        row = np.zeros(10)
        row[value] = 1
        mat.append(row)
    return np.array(mat)

def load_checking_images():
    file_data = gzip.open('data/check_data.gz', 'rb')
    data = np.array(list(file_data.read()))
    return np.divide(data[16:], 255.0)

def load_checking_labels():
    file_data = gzip.open('data/check_labels.gz', 'rb')
    labels = file_data.read()
    mat = []
    for value in labels[8:]:
        row = np.zeros(10)
        row[value] = 1
        mat.append(row)
    return np.array(mat)