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


def encode_inchi_name(inchi_name: str, codex_list: list):
    """
    This function encodes an InChI identifier using One-Hot Encoding and floating point numbers

    :param inchi_name: InChI Identifier string
    :param codex_list: List of all character values encountered in the InChI identifiers
    :return: encoded_name: Encoded version of the InChI identifier
    """
    encoded_name = []
    encoded_numeric = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    for character in inchi_name:
        if character.isnumeric():
            encoded_name.append([encoded_numeric, [float(character)]])
        else:
            char_index = codex.index(character)
            encoded_character = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            encoded_character[char_index] = 1.0
            encoded_name.append([encoded_character, [0.0]])

    return encoded_name


def decode_inchi_name(encoded_name: list, codex_list: list):
    """
    This function decodes the One-Hot Encoded and floating point numbers back to the original InChI identifier

    :param encoded_name: Encoded version of the InChI identifier
    :param codex_list: List of all character values encountered in the InChI identifiers
    :return: inchi_name: InChI Identifier string
    """
    inchi_name = ''

    for encoded_character in encoded_name:
        if encoded_character[0][-1] == 1.0:
            inchi_name = inchi_name + str(int(encoded_character[1][0]))
        else:
            inchi_name = inchi_name + codex_list[encoded_character[0].index(1.0)]

    return inchi_name


def data_generator(labels: list, folder_options: list):
    """
    This generator provides the pre-processed image inputs for the model to use, as well as the input image's name and
    output InChI string.

    :return: image_data_array, image_name, output_string
    """
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

                for label in labels:
                    if label[0] == image_name:
                        output_string = label[1]
                        break

                yield image_data_array, image_name, output_string


if __name__ == '__main__':
    """
    """

    codex = []
    with open('Codex.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            codex = row

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
    folder_permutations = list(itertools.permutations(folder_layers, 3))

    train_gen = data_generator(training_labels, folder_permutations)
    validation_gen = data_generator(training_labels, folder_permutations)
    test_gen = data_generator(training_labels, folder_permutations)
    num_epochs = 100

    for i, training_data in enumerate(train_gen):
        if i > num_epochs:
            break
