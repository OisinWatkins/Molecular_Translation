import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import numpy as np

import logging

logging.getLogger('tensorflow').disabled = True

from PIL import Image, ImageOps
from os import listdir
from os.path import isfile, join
from tensorflow.keras import models
from molecular_translation import decode_inchi_name


def test_data_generator(test_data_dir: str = 'D:\\Datasets\\bms-molecular-translation\\test'):
    """
    Generator for the test dataset. Iterates through the entire test dataset and returns the images one by one.

    :param test_data_dir: Directory for the test dataset
    :return: image_data_array: Image data for a single image stored under one directory
    """
    folder_list = [folder for folder in listdir(test_data_dir)]
    for folder_option in folder_list:
        for subfolder_option_1 in folder_list:
            for subfolder_option_2 in folder_list:
                full_path = test_data_dir + '\\' + folder_option + '\\' + subfolder_option_1 + '\\' + subfolder_option_2 + '\\'
                file_list = [f for f in listdir(full_path) if isfile(join(full_path, f))]

                for file in file_list:
                    image_name = file[0:-4]

                    file_path = full_path + file
                    image_data = Image.open(file_path)
                    image_data = ImageOps.invert(image_data)
                    bg_colour = 0

                    image_data = image_data.convert('1')
                    image_data = ImageOps.pad(image_data, (1500, 1500), color=bg_colour)
                    image_data_array = np.array(image_data).astype(np.float32).reshape((1, 1500, 1500, 1))

                    yield image_name, image_data_array


if __name__ == '__main__':
    """
    """

    str_padding_len = 300

    # Grab the codex for the One-Hot Encoding scheme used here
    codex = []
    with open('Codex.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            codex = row
    codex_len = len(codex)
    print("\n-Codex has been loaded")

    data_gen = test_data_generator()

    model_directory = 'D:\\AI Projects\\Molecular_Translation Models\\GAN\\Gen - LSTM 1024   Dis - LSTM 1024   LR ' \
                      '1e-3   Decay 1e-8\\InChI_Generator_Checkpoint.h5 '

    generator_model = models.load_model(model_directory)

    while True:
        test_data = next(data_gen, None)
        if test_data is not None:
            test_image_name = test_data[0]
            test_image_data = test_data[1]

            test_prediction = generator_model.predict(x=test_image_data)
            test_str_prediction = np.round(test_prediction[0]).reshape((str_padding_len, len(codex)+1))
            test_num_prediction = np.round(test_prediction[1]).reshape((str_padding_len, 1))

            total_prediction = [test_str_prediction, test_num_prediction]
            predicted_inchi = decode_inchi_name(total_prediction, codex)

            test_output = [test_image_name, predicted_inchi]

            print(f"Image Name: {test_output[0]} --> Predicted InChI: {test_output[1]}")

        else:
            break
