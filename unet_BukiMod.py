# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:41:15 2022

@author: 
    Buki: olubukola.ishola@okstate.edu
    Data and Computational Geoscientist

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
"""
"""
code requirement(s):
    install tensorflow:  "pip install tensorflow" see more information here: https://www.tensorflow.org/
    install keras:  "pip install keras" see more information here: https://pypi.org/project/keras/
"""

#Loading libraries...
import tensorflow as tf
import numpy as np
import sys

#Contracting path, see https://keras.io/api/layers/
def contract(inputt, dropout, filters, activation):
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(inputt)
    c = tf.keras.layers.Dropout(dropout)(c)
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(c)
        
    return c

#Expanding path, see https://keras.io/api/layers/
def expand(inputt, dropout, filters, activation, concat):
    u = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputt)
    u = tf.keras.layers.concatenate([u, concat])
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(u)
    c = tf.keras.layers.Dropout(dropout)(c)
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(c)
      
    return c

#Expanding path for final output, see https://keras.io/api/layers/
def expand2(inputt, dropout, filters, activation, concat):
    u = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputt)
    u = tf.keras.layers.concatenate([u, concat], axis=3)
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(u)
    c = tf.keras.layers.Dropout(dropout)(c)
    c = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(c)
      
    return c

#Training algorithm
def fit(X_train, Y_train, model_name='model', activation='relu', optimizer='adam', patience=5, validation_split=0.3, batch_size=100, epochs=50, dropout=0.2, filters=64, depth=4, seed=1992):

    '''
    Description:
        Trains the designed algorithm with X_train and Y_train.
    
    Parameters: 
        Note: Default arguments can be seen in the arguments passed into the function and they are the default architecture is the standard unet number of depth and filters
        X_train (Array of uint8): 3D array of raw image with [thickness, height, width]
        Y_train (Array of uint8): 3D array of true segmentation with [thickness, height, width]
        model_name (string): Intended  name for saving model'.h5'
        activation (string): Intended activation functions, see https://keras.io/api/layers/activations/
        optimizer (string): Intended optimizer functions, see https://keras.io/api/optimizers/
        epochs (int): Number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        validation_split (float): Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.
        batch_size (int): Number of samples per gradient update. If unspecified, batch_size will default to 32.
        epochs (int): Number of epochs to train the model.
        seed (int): For repeatability, seed is set for random number generator.
        depth (int): Number of contactions on the contracting path and expansion.
        dropout (float): Float between 0 and 1. Fraction of the input units to drop.
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution). See https://keras.io/api/layers/convolution_layers/convolution2d/

    Returns:
        Saves model on computer.
        Returns history, a variable that holds the changes of binary cross entropy and accuracy with epoch 
    '''    

    #Setting random seed 
    np.random.seed = seed
    
    #Expanding dimenions, basically adding channel of one indicating its grayscale image
    X_train = np.expand_dims(X_train,3)
    Y_train = np.expand_dims(Y_train,3)
    Y_train = Y_train !=0
    
    #Reading data dimensions
    IMG_WIDTH = X_train.shape[1]
    IMG_HEIGHT = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    
    #Ensuring image has the same height and width
    if  IMG_WIDTH != IMG_HEIGHT:
        print('Error:::::::::::::Shape of each 2D image should be equal in height and width')
        print('Error:::::::::::::Consider padding image or cropping image along the height and width')
        print('Error:::::::::::::height of images should be divisible by 2 to the power (depth of unet)')
        print('Error:::::::::::::for example if "depth of unet" is 5 image height by 2^5 or 32')
        sys.exit()
    
    #Ensuring image does not run out of pixels from multiple maxpooling
    if  IMG_HEIGHT % 2**depth !=0:
        print('Error:::::::::::::height of images should be divisible by 2 to the power (depth of unet)')
        print('Error:::::::::::::for example if "depth of unet" is 5 image height by 2^5 or 32')
        sys.exit()

    #Building the U-net architecture
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
   
    #Contracting path
    c1 = contract(s, dropout, filters, activation)
    p = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    filters = filters*2 
    if depth > 1:
        c2 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        filters = filters*2 
    if depth > 2:
        c3 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        filters = filters*2 
    if depth > 3:
        c4 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        filters = filters*2 
    if depth > 4:
        c5 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c5)
        filters = filters*2 
    if depth > 5:
        c6 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c6)
        filters = filters*2 
    if depth > 6:
        c7 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c7)
        filters = filters*2 
    if depth > 7:
        c8 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c8)
        filters = filters*2 
    if depth > 8:
        c9 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c9)
        filters = filters*2 
    if depth > 9:
        c10 = contract(p, dropout, filters, activation)
        p = tf.keras.layers.MaxPooling2D((2, 2))(c10)
        filters = filters*2 

    #Base layer
    e = contract(p, dropout, filters, activation)
       
    #Expanding path 
    if depth > 9:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c10)
    if depth > 8:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c9)
    if depth > 7:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c8)
    if depth > 6:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c7)
    if depth > 5:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c6)
    if depth > 4:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c5)
    if depth > 3:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c4)
    if depth > 2:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c3)
    if depth > 1:
        filters = filters/2
        e = expand(e, dropout, filters, activation, c2)
    filters = filters/2
    e = expand2(e, dropout, filters, activation, c1)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(e)

    #Putting the model together     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    #Training the model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_name+'.h5', verbose=1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor = 'val_loss'), tf.keras.callbacks.TensorBoard('logs'), checkpointer]
    results = model.fit(X_train, Y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    
    return model.save, results

#Predicting algorithm
def predict(X_test, model_name):
    '''
    Description:
        This function segments X_train.
    
    Parameters: 
        Note: Default arguments can be seen inthe arguments passed into fit
        X_test (Array of uint8): 3D array of raw image with [thickness, height, width]
        model_name (string): The name of the model, do not add file extension, it automatically adds '.h5'
 
    Returns:
        Y_predict (Array of uint8): 3D array of predicted segmentation with [thickness, height, width]
    '''    
    X_test = np.expand_dims(X_test,3)
    model = tf.keras.models.load_model(model_name+'.h5')
    Y_predict = model.predict(X_test, verbose=1)
    Y_predict = np.squeeze(Y_predict,3)
    Y_predict = Y_predict*255
    Y_predict = Y_predict.astype(np.uint8)
    return Y_predict