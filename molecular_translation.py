import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import nltk
import time
import random
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging

logging.getLogger('tensorflow').disabled = True

from PIL import Image, ImageOps
from IPython import display
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers

"""
------------------------------------------------------------------------------------------------------------------------
Encoding and Decoding functions for InChI Names. Data Generator for general use.
------------------------------------------------------------------------------------------------------------------------
"""


def encode_inchi_name(inchi_name: str, codex_list: list, padded_size: int = 300):
    """
    This function encodes an InChI identifier using One-Hot Encoding and floating point numbers

    :param inchi_name: InChI Identifier string
    :param codex_list: List of all character values encountered in the InChI Identifiers
    :param padded_size: The required padded length of the encoded InChI Identifier
    :return: encoded_name: Encoded version of the InChI identifier
    """
    # Empty list for the encoded name
    encoded_name = []

    # Value encoding an empty part of the string (used for padding to the end)
    encoded_empty = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Value encoding a numeric part of the input string
    encoded_numeric = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]

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
            for char_after in inchi_name[index + 1:-1]:
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
            encoded_character = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                 -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            encoded_character[char_index] = 1.0
            encoded_name.append([encoded_character, [0.0]])

    # No pad the encoded InChI identifier with empty characters
    if len(encoded_name) < padded_size:
        for extra_i in range(padded_size - len(encoded_name)):
            encoded_name.append([encoded_empty, [0.0]])

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

    # Value encoding an empty part of the string (used for padding to the end)
    encoded_empty = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # For every encoded value in the list
    for encoded_character in encoded_name:
        # If the value encountered is a numeric value, concatenate the output string
        if encoded_character[0][-1] == 1.0:
            inchi_name = inchi_name + str(int(encoded_character[1][0]))
        # If the encoded character is empty, skip it
        elif encoded_character[0] == encoded_empty:
            continue
        # Otherwise concatenate the value in the codex at that given index
        else:
            try:
                # Attempt to decode the value provided
                inchi_name = inchi_name + codex_list[encoded_character[0].index(1.0)]
            except:
                # If it doesn't work, move on
                continue

    return inchi_name


def data_generator(labels: list, folder_options: list, codex_list: list, padded_size: int = 300, batch_size: int = 1,
                   dataset_path: str = 'D:\\Datasets\\bms-molecular-translation\\train\\',
                   folder_loop: int = 1, augment_data: bool = True, invert_image: bool = True, repeat_image: int = 1):
    """
    This generator provides the pre-processed image inputs for the model to use, as well as the input image's name and
    output InChI string.

    :return: image_data_array, image_name, output_string
    """
    # Limitations on the Augmentation performed on the training and validation inputs
    translation_mag = 10
    rotations_mag = 180

    while True:
        # Shuffle the folder order
        random.shuffle(folder_options)

        # Iterate through all folder paths
        for folder_path in folder_options:
            # Grab all files under a particular folder path
            full_path = dataset_path + folder_path[0] + '\\' + folder_path[1] + '\\' + folder_path[2] + '\\'
            file_list = [f for f in listdir(full_path) if isfile(join(full_path, f))]

            # Re-iterate over the same folder, shuffling the order each time
            for folder_itr in range(folder_loop):
                random.shuffle(file_list)

                # Iterate through each file, preprocess and yield each
                for file in file_list:
                    # Repeat each training input as many times as desired
                    for repeat in range(repeat_image):

                        # Instantiate the batch
                        image_data_batch = np.zeros(shape=(batch_size, 1500, 1500, 1))
                        output_str_batch = np.zeros(shape=(batch_size, padded_size, len(codex_list) + 1))
                        output_num_batch = np.zeros(shape=(batch_size, padded_size, 1))

                        # Generate as big a batch of data as is requested
                        for batch_num in range(batch_size):

                            # Prepare Image augmentations
                            rand_trans_mag_vert = round(np.random.uniform(-translation_mag, translation_mag))
                            rand_trans_mag_horizontal = round(np.random.uniform(-translation_mag, translation_mag))
                            rand_rotation = np.random.uniform(-rotations_mag, rotations_mag)

                            # Load image in Black and White with a constant size of 1500 x 1500
                            file_path = full_path + file
                            image_data = Image.open(file_path)

                            bg_colour = 1

                            if invert_image:
                                # Invert image colour
                                image_data = ImageOps.invert(image_data)
                                bg_colour = 0

                            image_data = image_data.convert('1')

                            if augment_data:
                                # Perform Augmentation
                                image_data = image_data.rotate(angle=rand_rotation,
                                                               translate=(rand_trans_mag_vert, rand_trans_mag_horizontal),
                                                               fillcolor=bg_colour,
                                                               expand=True)

                            image_data = ImageOps.pad(image_data, (1500, 1500), color=bg_colour)
                            image_data_array = np.array(image_data).astype(np.float32).reshape((1, 1500, 1500, 1))

                            # Find the correct label from the csv file data
                            image_name = file[0:-4]
                            output_string = ''
                            for label in labels:
                                if label[0] == image_name:
                                    output_string = label[1]
                                    break

                            output_encoded = encode_inchi_name(output_string, codex_list, padded_size)

                            # Extract all encoded Str and Num information separately.
                            output_str = []
                            output_num = []
                            for char in output_encoded:
                                output_str.append(char[0])
                                output_num.append(char[1])

                            # Cast Output Str and Num data to Numpy arrays and reshape to suit
                            output_str_encoded = np.array(output_str).reshape((1, padded_size, len(codex_list) + 1))
                            output_num_encoded = np.array(output_num).reshape((1, padded_size, 1))

                            # Add new data to batch
                            image_data_batch[batch_num] = image_data_array
                            output_str_batch[batch_num] = output_str_encoded
                            output_num_batch[batch_num] = output_num_encoded

                        yield image_data_batch, [output_str_batch, output_num_batch]  # , image_name


def levenshtein_distance(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return nltk.edit_distance(s1=decode_inchi_name(y_true, codex), s2=decode_inchi_name(y_pred, codex))


def progbar(curr, total, full_progbar, curr_presentation_num=None, total_presentations=None, loss_val_1=None, loss_val_2=None):
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    if curr_presentation_num is not None and total_presentations is not None and loss_val_1 is not None and loss_val_2 is not None:
        print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac),
              f'[Current Presentation: {curr_presentation_num}/{total_presentations}]', '[Str Loss Value: {:>7.2}]'.format(loss_val_1), '[Numeric Loss Val: {:>7.2}]'.format(loss_val_2), end='')
    else:
        print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')


def tf_shuffle_axis(value, axis=0, seed=None, name=None):
    """
    Shuffles the value of a Layer Tensor along the specified axis.

    :param value: TensorFlow layer output
    :param axis: Axis to be shuffled along
    :param seed: Seed value for tf.shuffle
    :param name: Name of the tf.shuffle operation
    :return: The shuffled TensorFlow layer output
    """
    perm = list(range(len(K.int_shape(value))))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm), seed=seed, name=name)
    value = tf.transpose(value, perm=perm)
    return value


"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

"""
------------------------------------------------------------------------------------------------------------------------
Code needed to build and train CVAE
Source:
    https://www.tensorflow.org/tutorials/generative/cvae
------------------------------------------------------------------------------------------------------------------------
"""


class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder.
    """

    def __init__(self, latent_dim, input_shape):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        encoding_input = keras.Input(shape=input_shape)
        encoding_layer = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same',
                                       activation='relu')(encoding_input)
        encoding_layer = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same',
                                       activation='relu')(encoding_layer)
        encoding_layer = layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same',
                                       activation='relu')(encoding_layer)
        encoding_layer = layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same',
                                       activation='relu')(encoding_layer)
        encoding_layer = layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same',
                                       activation='relu')(encoding_layer)
        encoding_layer = layers.Dense(1024, activation='relu')(encoding_layer)

        shape_before_flattening = K.int_shape(encoding_layer)
        encoder_flatten = layers.Flatten()(encoding_layer)
        # No activation
        encoding_output = layers.Dense(latent_dim + latent_dim)(encoder_flatten)
        self.encoder = models.Model(encoding_input, encoding_output, name='Encoder')

        decoder_input = keras.Input(shape=(latent_dim,))
        decoder_layer = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation=tf.nn.relu)(decoder_input)
        decoder_layer = layers.Reshape(target_shape=shape_before_flattening[1:])(decoder_layer)
        decoder_layer = layers.Dense(1024, activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=1024, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        # No activation
        decoder_output = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(decoder_layer)
        decoder_output = layers.Cropping2D(cropping=((2, 2), (2, 2)))(decoder_output)
        self.decoder = models.Model(decoder_input, decoder_output, name='Decoder')

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def build_and_train_cvae(lr=1e-4, latent_dimension=250, input_dimension=(1500, 1500, 1)):
    # First, let's build a CVAE to handle feature extraction
    optimizer = tf.keras.optimizers.Adam(lr)

    # Instantiate CVAE Model
    cvae_model = CVAE(latent_dimension, input_dimension)

    # Provide Summary for Encoder and Decoder Models
    print("\n\n")
    cvae_model.encoder.summary()
    print("\n\n")
    cvae_model.decoder.summary()
    print("\n\n")

    return cvae_model


"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

"""
------------------------------------------------------------------------------------------------------------------------
This section is devoted to the InChI Text Generation model
------------------------------------------------------------------------------------------------------------------------
"""


def build_text_gen(len_encoded_str, len_padded_str=300, lr=1e-4):
    """

    :param len_encoded_str:
    :param len_padded_str:
    :param lr:
    :return:
    """

    # First: let's build the Image Processing head of the model
    image_input_dimension = (1500, 1500, 1)
    image_processing_head_input = keras.Input(shape=image_input_dimension)
    image_processing_head = layers.SeparableConv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Image_Processing_Conv2D_1')(image_processing_head_input)
    image_processing_head = layers.Dropout(0.1, name='Image_Processing_Dropout_1')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Image_Processing_Conv2D_2')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Image_Processing_Dropout_2')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Image_Processing_Conv2D_3')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Image_Processing_Dropout_3')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=512, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Image_Processing_Conv2D_4')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Image_Processing_Dropout_4')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=len_padded_str, kernel_size=3,
                                                   strides=(2, 2), activation='relu',
                                                   name='Image_Processing_Conv2D_5')(image_processing_head)

    image_processed_shape = K.int_shape(image_processing_head)
    image_processing_head = layers.Reshape(target_shape=(image_processed_shape[1]*image_processed_shape[2],
                                                         image_processed_shape[3]))(image_processing_head)

    image_processing_head = tf.transpose(image_processing_head, perm=[0, 2, 1])

    image_processing_head = layers.LSTM(units=512, return_sequences=True,
                                        name='Image_Processing_LSTM_1')(image_processing_head)

    # Fifth: Join outputs from the input heads and process into encoded strings
    combined_input_processed = layers.LSTM(units=512, return_sequences=True, dropout=0.1,
                                           name='Combined_Input_LSTM_1')(image_processing_head)
    combined_input_processed = layers.LSTM(units=512, return_sequences=True, dropout=0.1,
                                           name='Combined_Input_LSTM_2')(combined_input_processed)
    combined_input_processed = layers.LSTM(units=512, return_sequences=True, dropout=0.1,
                                           name='Combined_Input_LSTM_3')(combined_input_processed)
    combined_input_processed = layers.LSTM(units=512, return_sequences=True, dropout=0.1,
                                           name='Combined_Input_LSTM_4')(combined_input_processed)

    # Sixth: Define each output tail and compile the model
    inchi_name_output_str = layers.LSTM(units=len_encoded_str, activation='tanh', return_sequences=True,
                                        name='InChI_Name_Str_Processing_LSTM')(combined_input_processed)

    inchi_name_output_num = layers.LSTM(units=1, activation=None, return_sequences=True,
                                        name='InChI_Name_Num_Processing_LSTM')(combined_input_processed)

    inchi_name_model = models.Model(inputs=image_processing_head_input,
                                    outputs=[inchi_name_output_str, inchi_name_output_num], name="InChI_Name_Generator")

    optimizer = tf.keras.optimizers.Adam(lr)
    losses = {
        'InChI_Name_Str_Processing_LSTM': tf.losses.MeanSquaredError(),
        'InChI_Name_Num_Processing_LSTM': tf.losses.MeanSquaredError()
    }
    losses_weights = {"InChI_Name_Str_Processing_LSTM": 1.0, "InChI_Name_Num_Processing_LSTM": 0.01}

    inchi_name_model.compile(optimizer=optimizer, loss=losses, loss_weights=losses_weights, metrics=[levenshtein_distance])

    print("\n\n")
    inchi_name_model.summary()
    print("\n\n")

    return inchi_name_model


def build_discriminator(len_encoded_str, len_padded_str=300, lr=1e-4):
    """

    :param len_encoded_str:
    :param len_padded_str:
    :param lr:
    :return:
    """
    # First: let's build the Image Processing head of the model
    image_input_dimension = (1500, 1500, 1)
    image_processing_head_input = keras.Input(shape=image_input_dimension)
    image_processing_head = layers.SeparableConv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Discr_Image_Processing_Conv2D_1')(image_processing_head_input)
    image_processing_head = layers.Dropout(0.1, name='Discr_Image_Processing_Dropout_1')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Discr_Image_Processing_Conv2D_2')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Discr_Image_Processing_Dropout_2')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Discr_Image_Processing_Conv2D_3')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Discr_Image_Processing_Dropout_3')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=512, kernel_size=3, strides=(2, 2), activation='relu',
                                                   name='Discr_Image_Processing_Conv2D_4')(image_processing_head)
    image_processing_head = layers.Dropout(0.1, name='Discr_Image_Processing_Dropout_4')(image_processing_head)
    image_processing_head = layers.SeparableConv2D(filters=len_padded_str, kernel_size=3,
                                                   strides=(2, 2), activation='relu',
                                                   name='Discr_Image_Processing_Conv2D_5')(image_processing_head)

    image_processed_shape = K.int_shape(image_processing_head)
    image_processing_head = layers.Reshape(target_shape=(image_processed_shape[1]*image_processed_shape[2],
                                                         image_processed_shape[3]))(image_processing_head)

    image_processing_head = tf.transpose(image_processing_head, perm=[0, 2, 1])
    # image_processing_head = tf_shuffle_axis(image_processing_head, axis=1)

    image_processing_head = layers.LSTM(units=1024, return_sequences=True,
                                        name='Discr_Image_Processing_LSTM_1')(image_processing_head)

    # Second: let's build the Encoded String input handling head
    str_input_dimension = (len_padded_str, len_encoded_str)
    str_processing_head_input = keras.Input(shape=str_input_dimension)

    # Third: let's build the Encoded Number input handling head
    num_input_dimension = (len_padded_str, 1)
    num_processing_head_input = keras.Input(shape=num_input_dimension)

    # Fourth: Concatenate the String and Number processed outputs
    combined_name_input = tf.concat([str_processing_head_input, num_processing_head_input], -1)
    combined_name_processed = layers.LSTM(units=1024, return_sequences=True,
                                          name='Discr_Combined_Name_LSTM_1')(combined_name_input)

    # Fifth: Join outputs from the input heads and process into encoded strings
    combined_input = tf.concat([image_processing_head, combined_name_processed], -1)
    combined_input_processed = layers.LSTM(units=1024, return_sequences=True,
                                           name='Discr_Combined_Input_LSTM_1')(combined_input)
    combined_input_processed = layers.Dropout(0.1, name='Discr_Combined_Input_Dropout_1')(combined_input_processed)
    combined_input_processed = layers.LSTM(units=1024, return_sequences=True,
                                           name='Discr_Combined_Input_LSTM_2')(combined_input_processed)
    combined_input_processed = layers.Dropout(0.1, name='Discr_Combined_Input_Dropout_2')(combined_input_processed)
    combined_input_processed = layers.LSTM(units=1024, return_sequences=True,
                                           name='Discr_Combined_Input_LSTM_3')(combined_input_processed)
    combined_input_processed = layers.Dropout(0.1, name='Discr_Combined_Input_Dropout_3')(combined_input_processed)
    combined_input_processed = layers.LSTM(units=1024, return_sequences=True,
                                           name='Discr_Combined_Input_LSTM_4')(combined_input_processed)

    # Sixth: Define each output tail and compile the model
    discriminator_output = layers.Flatten(name='Discriminator_Flatten')(combined_input_processed)
    discriminator_output = layers.Dropout(0.4, name='Discriminator_Dropout')(discriminator_output)
    discriminator_output = layers.Dense(units=1, name='Discriminator_Dense',
                                        activation='sigmoid')(discriminator_output)

    discriminator_model = models.Model(inputs=[image_processing_head_input, str_processing_head_input, num_processing_head_input],
                                       outputs=discriminator_output, name="InChI_Name_Discriminator")

    optimizer = tf.keras.optimizers.RMSprop(lr, clipvalue=1.0)
    losses = {
        'Discriminator_Dense': tf.losses.BinaryCrossentropy()
    }
    discriminator_model.compile(optimizer=optimizer, loss=losses)
    discriminator_model.trainable = False

    print("\n\n")
    discriminator_model.summary()
    print("\n\n")

    return discriminator_model


"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    """
    Main Function
    Here all data needed for this application will be loaded and prepared for use.
    All necessary generators will be instantiated prior to any development or training of any model.
    """

    print("\n\n\n-----Preparing to train on the bms_molecular_translation dataset-----")
    # Grab the codex for the One-Hot Encoding scheme used here
    codex = []
    with open('Codex.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            codex = row
    codex_len = len(codex)
    print("\n-Codex has been loaded")

    # Grab all Training Labels used for this dataset
    training_labels_file = 'D:\\Datasets\\bms-molecular-translation\\train_labels.csv'
    training_labels = []
    with open(training_labels_file, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if not idx == 0:
                training_labels.append(row)
    print("\n-Training Labels have been loaded")

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
    print("\n-Training Permutations ready")

    # Prepare all permutations of folders for the validation data. All folders under the uppermost 'e' folder are used
    validation_folder_layers = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8',
                                '9', '9', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
    validation_folder_permutations = list(itertools.permutations(training_folder_layers, 2))
    validation_folder_permutations = list(dict.fromkeys(validation_folder_permutations))
    for i in range(len(validation_folder_permutations)):
        validation_folder_permutations[i] = ['e', validation_folder_permutations[i][0],
                                             validation_folder_permutations[i][1]]
    print("\n-Validation Permutations ready")

    # Prepare all permutations of folders for the testing data. All folders under the uppermost 'f' folder are used
    testing_folder_layers = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '9',
                             '9', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
    testing_folder_permutations = list(itertools.permutations(training_folder_layers, 2))
    testing_folder_permutations = list(dict.fromkeys(testing_folder_permutations))
    for i in range(len(testing_folder_permutations)):
        testing_folder_permutations[i] = ['f', testing_folder_permutations[i][0], testing_folder_permutations[i][1]]
    print("\n-Testing Permutations ready")

    str_padding_len = 300
    num_repeat_image = 1
    batch_length = 2

    # Instantiate all generators needed for training, validation and testing
    train_gen = data_generator(training_labels, training_folder_permutations, codex, batch_size=batch_length,
                               padded_size=str_padding_len, repeat_image=num_repeat_image)

    validation_gen = data_generator(training_labels, validation_folder_permutations, codex, padded_size=str_padding_len)

    test_gen = data_generator(training_labels, testing_folder_permutations, codex,
                              padded_size=str_padding_len, augment_data=False)
    print("\n-Data Generators are ready")

    """
    --------------------------------------------------------------------------------------------------------------------
    Now building and training CVAE
    --------------------------------------------------------------------------------------------------------------------
    """
    # cvae_model = build_and_train_cvae()

    # # Train Model according to the hyperparameters defines above
    # print("-----Beginning Training-----")
    # loss = tf.keras.metrics.Mean()
    #
    # for epoch in range(1, epochs + 1):
    #     start_time = time.time()
    #     print(f"Epoch: {epoch} Training:")
    #     for presentation_num, train_x in enumerate(train_gen):
    #         if presentation_num == presentations:
    #             break
    #         train_loss = train_step(cvae_model, train_x[0], optimizer)
    #         progbar(presentation_num, presentations, 20, train_loss)
    #         cvae_model.encoder.save('Encoder_training.h5', overwrite=True)
    #         cvae_model.decoder.save('Decoder_training.h5', overwrite=True)
    #     end_time = time.time()
    #
    #     for presentation_num, val_x in enumerate(validation_gen):
    #         if presentation_num == (presentations / 10):
    #             break
    #         loss(compute_loss(cvae_model, val_x[0]))
    #     elbo = -loss.result()
    #     display.clear_output(wait=False)
    #     print(f"\tValidation set ELBO: {elbo}\tTime elapse for current epoch: {end_time - start_time}")
    #
    # # Save each model before continuing
    # cvae_model.encoder.save('Encoder.h5')
    # cvae_model.decoder.save('Decoder.h5')
    """
    --------------------------------------------------------------------------------------------------------------------
    CVAE Built and Saved
    --------------------------------------------------------------------------------------------------------------------
    """

    """
    --------------------------------------------------------------------------------------------------------------------
    Now building and training InChI String Generation Model
    --------------------------------------------------------------------------------------------------------------------
    """
    inchi_model = build_text_gen(len_encoded_str=codex_len+1, len_padded_str=str_padding_len, lr=1e-4)
    # inchi_discriminator = build_discriminator(len_encoded_str=codex_len+1, len_padded_str=str_padding_len, lr=1e-4)

    checkpoint_filepath = 'D:\\AI Projects\\Molecular_Translation\\'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='InChI_Name_Str_Processing_LSTM_mse',
        mode='min',
        save_best_only=True)

    inchi_model.fit(x=train_gen, epochs=10000, steps_per_epoch=50, verbose=1, callbacks=[model_checkpoint_callback])

    inchi_model.save("InChI_Model.h5")

    """
    --------------------------------------------------------------------------------------------------------------------
    CVAE Built and Saved
    --------------------------------------------------------------------------------------------------------------------
    """
