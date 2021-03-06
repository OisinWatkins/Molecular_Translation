import csv
import random
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image


def encode_inchi_name(inchi_name: str = ''):
    return []


def decode_inchi_name(encoded_name=None):
    if encoded_name is None:
        encoded_name = []
    return ''


def training_generator():
    """
    This generator provides the pre-processed image inputs for the model to use, as well as the input image's name and
    output InChI string.
    
    :return: image_data_array, image_name, output_string
    """
    training_labels_file = 'D:\\Datasets\\bms-molecular-translation\\train_labels.csv'
    training_labels = []
    with open(training_labels_file, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if not idx == 0:
                training_labels.append(row)

    dataset_path = 'D:\\Datasets\\bms-molecular-translation\\train\\'
    folder_layers = ['0', '0', '0', '1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4', '5', '5', '5', '6', '6',
                     '6', '7', '7', '7', '8', '8', '9', '9', '9', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd',
                     'd', 'e', 'e', 'e', 'f', 'f', 'f']
    folder_options = list(itertools.permutations(folder_layers, 3))

    while True:
        random.shuffle(folder_options)

        for folder_path in folder_options:
            full_path = dataset_path + folder_path[0] + '\\' + folder_path[1] + '\\' + folder_path[2] + '\\'
            file_list = [f for f in listdir(full_path) if isfile(join(full_path, f))]

            for file in file_list:
                file_path = full_path + file
                image_data = Image.open(file_path)
                image_data = image_data.convert('1')
                image_data = ImageOps.pad(image_data, (1500, 1000), color=1)

                image_data_array = np.array(image_data).astype(np.float32)

                image_name = file[0:-4]
                output_string = ''

                for row in training_labels:
                    if row[0] == image_name:
                        output_string = row[1]
                        break

                yield image_data_array, image_name, output_string


if __name__ == '__main__':
    num_epochs = 100
    for i, training_data in enumerate(training_generator()):
        if i > num_epochs:
            break
