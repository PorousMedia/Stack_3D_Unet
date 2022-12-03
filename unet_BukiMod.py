# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:41:15 2022

@author: 
    Buki: olubukola.ishola@okstate.edu
    Machine learning and Computational Geoscientist

about:
    unet_BukiMod is a modification to the Unet architecture for semantic segmentation of images. 
    This is a useful tool in semantic segmentation of images.
    The modifications are:
        The ability to run Unet as a function without having to code the algorithm
        The ability to reduce the depth (contraction and corresponding expansion) of the Unet algorithm. A standrad Unet has a depth of 4, this modification allows varying depth of one to ten.
        The ability to adjust the number of filters in the first convolution layer. The default Unet has 16 filters. Lie Unet the number doubles along the contraction path and halfs along the expansion path.
    Note: 
        This code is developed for grayscale images so the number of channels is one. 

credit: 
    U-net: Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    DigitalSreeni: https://www.youtube.com/c/DigitalSreeni/

code requirement(s):
    install tensorflow:  "pip install tensorflow" see more information here: https://www.tensorflow.org/
    install keras:  "pip install keras" see more information here: https://pypi.org/project/keras/
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
import logging

logger = logging.getLogger(__name__)


def convolve(image, dropout, filters, activation, unet_3d):

    logging.debug('Convolving image')
    if not unet_3d:
        # Here, we convolve image with a filter  i.e convolution, see https://keras.io/api/layers/.
        image = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(3, 3),
                                       activation=activation,
                                       kernel_initializer='he_normal',
                                       padding='same')(image)
        image = tf.keras.layers.Dropout(dropout)(image)

        # Yes, another convolution
        image = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(3, 3),
                                       activation=activation,
                                       kernel_initializer='he_normal',
                                       padding='same')(image)

    if unet_3d:
        # Here, we convolve image with a filter  i.e convolution, see https://keras.io/api/layers/.
        image = tf.keras.layers.Conv3D(filters=filters,
                                       kernel_size=(3, 3, 3),
                                       activation=activation,
                                       kernel_initializer='he_normal',
                                       padding='same')(image)
        image = tf.keras.layers.Dropout(dropout)(image)

        # Yes, another convolution
        image = tf.keras.layers.Conv3D(filters=filters,
                                       kernel_size=(3, 3, 3),
                                       activation=activation,
                                       kernel_initializer='he_normal',
                                       padding='same')(image)
    return image


def contract(image, filters, unet_3d):

    logging.debug('Encoding image')
    if not unet_3d:
        # Here, we apply the pooling layer to the image i.e. contract the image.
        image = tf.keras.layers.MaxPooling2D((2, 2))(image)
        filters = filters * 2  # To help double the filter for the next layer if applicable

    if unet_3d:
        # Here, we apply the pooling layer to the image i.e. contract the image.
        image = tf.keras.layers.MaxPooling3D((2, 2, 2))(image)
        filters = filters * 2  # To help double the filter for the next layer if applicable
    return image, filters


# Decoder path
def expand(image, dropout, filters, activation, skip_connect_image, unet_3d):

    logging.debug('Decoding image')
    if not unet_3d:
        filters = filters / 2  # systematically restoring image to original shape

        # Upsampling, to negate the effect of pooling layert i.e expanding path
        image = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                kernel_size=(2, 2),
                                                strides=(2, 2),
                                                padding='same')(image)

        # Skip connection is happening here
        image = tf.keras.layers.concatenate([image, skip_connect_image])

        # Apply convolution.
        image = convolve(image, dropout, filters, activation, unet_3d)

    if unet_3d:
        filters = filters / 2  # systematically restoring image to original shape

        # Upsampling, to negate the effect of pooling layert i.e expanding path
        image = tf.keras.layers.Conv3DTranspose(filters=filters,
                                                kernel_size=(2, 2, 2),
                                                strides=(2, 2, 2),
                                                padding='same')(image)

        # Skip connection is happening here
        image = tf.keras.layers.concatenate([image, skip_connect_image])

        # Apply convolution.
        image = convolve(image, dropout, filters, activation, unet_3d)
    return image, filters


# The dice coefficient used in computing accuracy in ML model
def dice_coefficient(y_true, y_pred):

    logging.debug('Computing dice coefficient')
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (
        K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)


# The actual algorithm for training starts here
def fit(training_xray,
        training_mask,
        validation_xray,
        validation_mask,
        model_name='model',
        activation='relu',
        optimizer='adam',
        patience=3,
        batch_size=50,
        epochs=50,
        dropout=0.2,
        filters=16,
        depth=4,
        seed=1992,
        unet_3d=False):
    """
    Construct and trains specified unet algorithm.

    Parameters
    ----------
    training_xray : Array of float32
        4D array of raw image with [thickness, height, width, no_of_channels].
    training_mask : Array of float32
        4D array of true segmentation with [thickness, height, width, no_of_channels].
    validation_xray : Array of float32
        4D array of raw image with [thickness, height, width, no_of_channels].
    validation_mask : Array of float32
        4D array of true segmentation with [thickness, height, width, no_of_channels].
    model_name : string, optional
        The name of the model, do not add file extension, it automatically adds '.h5'. The default is 'model'.
    activation : string, optional
        Intended activation functions, see https://keras.io/api/layers/activations/. The default is 'relu'.
    optimizer : string, optional
        Intended optimizer functions, see https://keras.io/api/optimizers/. The default is 'adam'.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped. The default is 2.
    batch_size : int, optional
        Number of samples per gradient update. The default is 32.
    epochs : int, optional
        Number of epochs to train the model. The default is 50.
    dropout : float, optional
        Float between 0 and 1. Fraction of the input units to drop. The default is 0.2.
    filters : int, optional
        The dimensionality of the output space (i.e. the number of output filters in the convolution). See https://keras.io/api/layers/convolution_layers/convolution2d/. The default is 64.
    depth : int, optional
        Number of contactions on the contracting path and expansion. The default is 4.
    seed : int, optional
        For repeatability, seed is set for random number generator. The default is 1992.

    Returns
    -------
    Saves model: 'h5'
        Saves model on computer.
    results : numpy array
        Returns history, a variable that holds the changes of binary cross entropy and accuracy with epoch.

    """

    # Note to self. Log available CPU and GPU memory available
    logging.info(
        f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")
    logging.info(
        f"Num CPUs Available: {len(tf.config.list_physical_devices('CPU'))}.")

    np.random.seed(seed)

    logging.debug('Deducing image dimensions.')
    if not unet_3d:
        img_width = training_xray.shape[1]
        img_height = training_xray.shape[2]
        img_channels = training_xray.shape[3]
        logging.info(
            f"Image has dimensions {[img_width, img_height, img_channels]}")

    if unet_3d:
        img_thickness = training_xray.shape[1]
        img_width = training_xray.shape[2]
        img_height = training_xray.shape[3]
        img_channels = training_xray.shape[4]
        logging.info(
            f"Image has dimensions {[img_thickness, img_width, img_height, img_channels]}"
        )

    # Ensuring image has the same height and width
    if not unet_3d:
        if img_width != img_height:
            raise ValueError(
                "Shape of each 2D image should be equal in height and width, consider padding image or cropping image along the height and width, height of images should be divisible by 2 to the power (depth of unet). Example if 'depth of unet' is 5 image height by 2^5 or 32"
            )

    if unet_3d:
        # Ensuring image has the same height and width
        if img_width != img_thickness:
            raise ValueError(
                "Shape of each 3D image should be equal in height, width and thickness, consider padding image or cropping image along the height, width and thickness, height of images should be divisible by 2 to the power (depth of unet). Example if 'depth of unet' is 5 image height by 2^5 or 32"
            )

    # Ensuring image does not run out of pixels from multiple maxpooling
    if img_height % 2**depth != 0:
        raise ValueError(
            "height of images should be divisible by 2 to the power (depth of unet), for example if 'depth of unet' is 5 image height by 2^5 or 32"
        )

    # Note to self. Revisit distributed training. Currently, more data than available GPU memory
    logging.debug("Enforcing Multi-worker configuration.")
    strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    logging.info(
        f"The number of workers available is {strategy.num_replicas_in_sync}.")
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    logging.debug("Formatting data with tensorflow tools.")
    train = tf.data.Dataset.from_tensor_slices(
        (training_xray, training_mask)).shuffle(seed).batch(global_batch_size)
    validation = tf.data.Dataset.from_tensor_slices(
        (validation_xray,
         validation_mask)).shuffle(seed).batch(global_batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train = train.with_options(options)
    validation = validation.with_options(options)

    logging.debug("Specifying input file size.")
    if not unet_3d:
        inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

    if unet_3d:
        inputs = tf.keras.layers.Input(
            (img_thickness, img_height, img_width, img_channels))

    logging.debug("Setting up contracting path.")
    conv_ = []
    conv = convolve(inputs, dropout, filters, activation, unet_3d)
    pool, filters = contract(conv, filters, unet_3d)
    conv_.append(conv)

    for _ in range(1, depth):
        conv = convolve(pool, dropout, filters, activation, unet_3d)
        pool, filters = contract(conv, filters, unet_3d)
        conv_.append(conv)

    logging.debug(
        "Floor of the U-net, after here, we can keep the variable name 'base' because we are not keeping any image for concatenation."
    )
    base = convolve(pool, dropout, filters, activation, unet_3d)

    logging.debug("Setting up expanding path.")
    for i in reversed(range(depth)):
        base, filters = expand(base, dropout, filters, activation, conv_[i],
                               unet_3d)

    logging.debug(
        "Output segmentation map. Note we are solving a binary problem in this code."
    )
    if not unet_3d:
        outputs = tf.keras.layers.Conv2D(filters=1,
                                         kernel_size=(1, 1),
                                         activation='sigmoid')(base)

    if unet_3d:
        outputs = tf.keras.layers.Conv3D(filters=1,
                                         kernel_size=(1, 1, 1),
                                         activation='sigmoid')(base)
    logging.debug("Putting the model together.")
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=dice_coefficient)
    print(model.summary())

    logging.debug("Training the model.")
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_name + '.h5',
                                                      verbose=1,
                                                      save_best_only=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience,
                                         monitor='val_loss'),
        tf.keras.callbacks.TensorBoard('logs'), checkpointer
    ]
    results = model.fit(train,
                        epochs=epochs,
                        validation_data=validation,
                        callbacks=callbacks)
    return results


# The  algorithm for segmenting the data starts here
def predict(image, model_name, unet_3d=False):
    """
    This function segments X_ray images (binary).

    Parameters
    ----------
    X_test : Array of float32
        4D array of raw image with [thickness, height, width, no_of_channels].
    model_name : string
        The name of the model, do not add file extension, it automatically adds '.h5'.

    Returns
    -------
    Y_predict : Array of uint8
        3D array of predicted segmentation with [thickness, height, width].

    """

    logging.debug("Loading model.")
    model = tf.keras.models.load_model(model_name + '.h5', compile=False)

    logging.debug(
        "Since we are expecting grayscale images, the dimension for the channel is removed after prediction."
    )
    if not unet_3d:
        segmented_image = model.predict(image, verbose=1)
        segmented_image = np.squeeze(segmented_image, 3)

    if unet_3d:
        segmented_image = model.predict(image, batch_size=1, verbose=1)
        segmented_image = np.squeeze(segmented_image, 4)

    logging.debug("Data is scaled to uint8.")
    segmented_image = segmented_image * 255
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image
