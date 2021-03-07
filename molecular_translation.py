import re
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
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image


def encode_inchi_name(inchi_name: str, codex_list: list):
    """
    This function encodes an InChI identifier using One-Hot Encoding and floating point numbers

    :param inchi_name: InChI Identifier string
    :param codex_list: List of all character values encountered in the InChI identifiers
    :return: encoded_name: Encoded version of the InChI identifier
    """
    # Empty list for the encoded name
    encoded_name = []

    # Value encoding a numeric part of the input string
    encoded_numeric = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Counter used to control skipping values in the string, specifically for numerics longer than 2 values
    counter = 0

    # Iterate through each character in the string, encoding each as either a string or numeric character
    for index, character in enumerate(inchi_name):
        # Skip any repeated numeric value
        if not counter == 0:
            counter = counter - 1
            continue
        # In a numeric character is encountered handle it here
        elif character.isnumeric():
            numeric_str = character
            # Handle numeric values longer than 1 character
            for char_after in inchi_name[index+1:-1]:
                if char_after.isnumeric():
                    numeric_str = numeric_str + char_after
                    counter = counter + 1
                else:
                    break

            # Add the encoded numeric and the floating point value to the encoded_name
            encoded_name.append([encoded_numeric, [float(numeric_str)]])

        # Otherwise encode a string value normally using the provided codex_list
        else:
            char_index = codex_list.index(character)
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
    # Empty string to store the decoded name
    inchi_name = ''

    # For every encoded value in the list
    for encoded_character in encoded_name:
        # If the value encountered is a numeric value, concatenate the output string
        if encoded_character[0][-1] == 1.0:
            inchi_name = inchi_name + str(int(encoded_character[1][0]))
        # Otherwise concatenate the value in the codex at that given index
        else:
            inchi_name = inchi_name + codex_list[encoded_character[0].index(1.0)]

    return inchi_name


def data_generator(labels: list, folder_options: list, dataset_path: str = 'D:\\Datasets\\bms-molecular-translation'
                                                                           '\\train\\'):
    """
    This generator provides the pre-processed image inputs for the model to use, as well as the input image's name and
    output InChI string.

    :return: image_data_array, image_name, output_string
    """
    while True:
        # Shuffle the folder order
        random.shuffle(folder_options)

        # Iterate through all folder paths
        for folder_path in folder_options:
            # Grab all files under a particular folder path
            full_path = dataset_path + folder_path[0] + '\\' + folder_path[1] + '\\' + folder_path[2] + '\\'
            file_list = [f for f in listdir(full_path) if isfile(join(full_path, f))]

            # Iterate through each file, preprocess and yield each
            for file in file_list:
                # Load image in Black and White with a constant size of 1500 x 1000
                file_path = full_path + file
                image_data = Image.open(file_path)
                image_data = image_data.convert('1')
                image_data = ImageOps.pad(image_data, (1500, 1000), color=1)

                image_data_array = np.array(image_data).astype(np.float32)

                # Find the correct label from the csv file data
                image_name = file[0:-4]
                output_string = ''
                for label in labels:
                    if label[0] == image_name:
                        output_string = label[1]
                        break

                yield image_data_array, output_string


if __name__ == '__main__':
    """
    Main Function
    Here all data needed for this application will be loaded and prepared for use.
    All necessary generators will be instantiated prior to any development or training of any model.
    """

    # Grab the codex for the One-Hot Encoding scheme used here
    codex = []
    with open('Codex.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            codex = row

    # Grab all Training Labels used for this dataset
    training_labels_file = 'D:\\Datasets\\bms-molecular-translation\\train_labels.csv'
    training_labels = []
    with open(training_labels_file, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if not idx == 0:
                training_labels.append(row)

    # Prepare all permutations of folders for the training data. All folders except the uppermost 'e' and 'f' folders
    # are used
    training_folder_layers = ['0', '0', '0', '1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4', '5', '5', '5',
                              '6', '6', '6', '7', '7', '7', '8', '8', '9', '9', '9', 'a', 'a', 'a', 'b', 'b', 'b', 'c',
                              'c', 'c', 'd', 'd', 'd', 'e', 'e', 'e', 'f', 'f', 'f']
    training_folder_permutations = list(itertools.permutations(training_folder_layers, 3))
    training_folder_permutations = list(dict.fromkeys(training_folder_permutations))
    for permutation in training_folder_permutations:
        if permutation[0] == 'e' or permutation[0] == 'f':
            training_folder_permutations.remove(permutation)

    # Prepare all permutations of folders for the validation data. All folders under the uppermost 'e' folder are used
    validation_folder_layers = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8',
                                '9', '9', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
    validation_folder_permutations = list(itertools.permutations(training_folder_layers, 2))
    validation_folder_permutations = list(dict.fromkeys(validation_folder_permutations))
    for i in range(len(validation_folder_permutations)):
        validation_folder_permutations[i] = ['e', validation_folder_permutations[i][0], validation_folder_permutations[i][1]]

    # Prepare all permutations of folders for the testing data. All folders under the uppermost 'f' folder are used
    testing_folder_layers = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4','5', '5', '6', '6', '7', '7', '8', '9',
                             '9', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
    testing_folder_permutations = list(itertools.permutations(training_folder_layers, 2))
    testing_folder_permutations = list(dict.fromkeys(testing_folder_permutations))
    for i in range(len(testing_folder_permutations)):
        testing_folder_permutations[i] = ['f', testing_folder_permutations[i][0], testing_folder_permutations[i][1]]

    # Instantiate all generators needed for training, validation and testing
    train_gen = data_generator(training_labels, training_folder_permutations)
    validation_gen = data_generator(training_labels, validation_folder_permutations)
    test_gen = data_generator(training_labels, testing_folder_permutations)

    # First, let's build a VAE to handle feature extraction
    img_shape = (1500, 1000, 1)
    latent_dim = 100

    input_img = keras.Input(shape=img_shape)

    # Now let's provide a collection of layers to handle encoding the image input.
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Finally let's encode the image input using Dense layers. Be sure to grab the shape of the Tensor
    # before flattening for use later.
    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    def sampling(args):
        """
        This function will encode and sample the encoding space. We'll use this function in a Lambda
        layer later.

        :param: args:
        :return: A sampled point in the encoded space.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)

        return z_mean + K.exp(z_log_var) * epsilon

    # Use the sampling function in a Lambda layer.
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Now let's make the decoder.
    decoder_input = layers.Input(K.int_shape(z)[1:])

    # Upsample the image.
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)

    # Reshape z to match the shape of z prior to flattening.
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # Use a Conv2DTranspose and a Conv2D to decode z into an output that's the same size as the input.
    x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    # Build the decoder model and define the decoder output.
    decoder = models.Model(decoder_input, x)
    z_decoded = decoder(z)

    # The dual loss of a VAE doesn't fit the tradional expectation of a sample-wise function of the normal format
    # loss(input, target). Thus we'll need to setup the loss by writing a custom layer that internally uses a
    # built-in add_loss layer method to create an arbitrary loss.
    class CustomVariationLayer(keras.layers.Layer):

        def vae_loss(self, x, z_decoded):
            """
            Custom loss function for VAE's

            :param: self:
            :param: x: Input image
            :param: z_decoded: Output image
            :return: Mean of xent_loss and kl_loss
            """
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs, **kwargs):
            """
            This function is used to implement the custom VAE loss function defined above. The output from the
            function is not needed by the caller, however the function is required to return something.

            :param: self: CustomVariationLayer object
            :param: inputs: A list containing both the input values and the decoded values.
            :return: Anything, the function output isn't used by the caller.
            """
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            return x

    y = CustomVariationLayer()([input_img, z_decoded])

    # Now let's compile and train the model.
    vae = models.Model(input_img, y)
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()
