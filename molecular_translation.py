import csv
import time
import random
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from IPython import display
from os import listdir
from os.path import isfile, join
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

                image_data_array = np.array(image_data).astype(np.float32).reshape((1, 1000, 1500, 1))

                # Find the correct label from the csv file data
                image_name = file[0:-4]
                output_string = ''
                for label in labels:
                    if label[0] == image_name:
                        output_string = label[1]
                        break

                yield image_data_array, None  # output_string


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

        encoding_input = layers.InputLayer(input_shape=input_shape)
        encoding_layer = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(encoding_input)
        encoding_layer = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(encoding_layer)
        encoding_layer = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu')(encoding_layer)

        shape_before_flattening = K.int_shape(encoding_layer)

        encoder_flatten = layers.Flatten()(encoding_layer)
        # No activation
        encoding_output = layers.Dense(latent_dim + latent_dim)(encoder_flatten)

        self.encoder = models.Model(encoding_input, encoding_output)

        decoder_input = layers.InputLayer(input_shape=(latent_dim,))
        decoder_layer = layers.Dense(units=np.prod(shape_before_flattening), activation=tf.nn.relu)(decoder_input)
        decoder_layer = layers.Reshape(target_shape=shape_before_flattening)(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        decoder_layer = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                               activation='relu')(decoder_layer)
        # No activation
        decoder_output = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(decoder_layer)

        self.decoder = models.Model(decoder_input, decoder_output)

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


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


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
        validation_folder_permutations[i] = ['e', validation_folder_permutations[i][0],
                                             validation_folder_permutations[i][1]]

    # Prepare all permutations of folders for the testing data. All folders under the uppermost 'f' folder are used
    testing_folder_layers = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '9',
                             '9', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
    testing_folder_permutations = list(itertools.permutations(training_folder_layers, 2))
    testing_folder_permutations = list(dict.fromkeys(testing_folder_permutations))
    for i in range(len(testing_folder_permutations)):
        testing_folder_permutations[i] = ['f', testing_folder_permutations[i][0], testing_folder_permutations[i][1]]

    # Instantiate all generators needed for training, validation and testing
    train_gen = data_generator(training_labels, training_folder_permutations)
    validation_gen = data_generator(training_labels, validation_folder_permutations)
    test_gen = data_generator(training_labels, testing_folder_permutations)

    """
    --------------------------------------------------------------------------------------------------------------------
    Now building and training CVAE
    --------------------------------------------------------------------------------------------------------------------
    """
    # First, let's build a VAE to handle feature extraction
    optimizer = tf.keras.optimizers.Adam(1e-4)

    epochs = 10
    presentations = 5
    latent_dimension = 2
    input_dimension = (1000, 1500, 1)
    num_examples_to_generate = 16

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dimension])
    cvae_model = CVAE(latent_dimension, input_dimension)

    # generate_and_save_images(cvae_model, 0, test_sample)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for presentation_num, train_x in enumerate(train_gen):
            if presentation_num == presentations:
                break
            train_step(cvae_model, train_x[0], optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for presentation_num, test_x in enumerate(test_gen):
            if presentation_num == presentations:
                break
            loss(compute_loss(cvae_model, test_x[0]))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

        # generate_and_save_images(cvae_model, epoch, test_sample)

    cvae_model.save('cvae_model.h5')

    """
    --------------------------------------------------------------------------------------------------------------------
    CVAE Built and Saved
    --------------------------------------------------------------------------------------------------------------------
    """
