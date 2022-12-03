#!/usr/bin/env python
__author__ = "Olubukola Ishola"

__email__ = "olubukola.ishola@okstate.edu"
"""
@author: Olubukola Ishola
"""

import patchify_buki as pt
import imageio as io
import numpy as np
import unet3D_buki_algo as ut
import tensorflow as tf
import gc
import logging
import os

logger = logging.getLogger(__name__)


def unet_buki_trainer(xray_image,
                      segmented_image,
                      image_patch=32,
                      validation_percentage=12.5,
                      training_epochs=50,
                      num_of_filters=16,
                      num_of_layers=4,
                      training_patience=3,
                      batch_size=100,
                      model_name="model",
                      random_seed=1992,
                      unet_3d=False):
    """
    This is just function to train a segmentation model using the unet_buki_algo and patchify_buki.   

    Parameters
    ----------
    xray_image : 3D array
        Xray data.
    segmented_image : 3D array
        Segmented version of xray_image.
    image_patch : int, optional
        The size of image sub-division. Note its a perfect square. The default is 32.
    validation_percentage : float, optional
        Percentage of data to use for training. The default is 30.
    training_epochs : int, optional
        Number of epochs for training. The default is 10.
    num_of_filters : int, optional
        Number of filters in the first layer. Note, it doubles with the layer depth. The default is 16.
    num_of_layers : int, optional
        Number of layers in the model. The default is 4.
    training_patience : int, optional
        This is the number of epoch after which the model training stops if no improvement in validation loss is observed. The default is 3.
    batch_size : int, optional
        DESCRIPTION. The default is 64.
    model_name : string, optional
        Name of the model. The default is "model".
    random_seed : int, optional
        A random seed is required for shuffling of data for it to generalize better. The default is 1992.
    unet_3d : bolean, optional
        True if a 3D CNN is intended, False for 2D. The default is False.
        
    Returns
    -------
    Saves model: 'h5'
        Saves model on computer.
    results : numpy array
        Returns history, a variable that holds the changes of binary cross entropy and accuracy with epoch.

    """

    if not unet_3d:
        img_depth = 1
    if unet_3d:
        img_depth = image_patch

    logging.debug("Formatting Xray.")
    xray_image = ((xray_image - np.min(xray_image)) * np.float32(255.0)) / (
        np.max(xray_image) - np.min(xray_image))

    logging.debug("Formatting Mask.")
    segmented_image = ((segmented_image - np.min(segmented_image)) *
                       np.float32(255.0)) / (np.max(segmented_image) -
                                             np.min(segmented_image))

    if image_patch:
        logging.debug("Patching Image data.")
        xray_image = pt.patch(xray_image, img_depth, image_patch, image_patch)
    xray_image = xray_image / np.float32(255)  # Note to self: deprecate

    if not unet_3d:
        xray_image = np.expand_dims(xray_image, 3)

    if unet_3d:
        xray_image = np.expand_dims(xray_image, 4)

    if image_patch:
        logging.debug("Patching Mask data.")
        segmented_image = pt.patch(segmented_image, img_depth, image_patch,
                                   image_patch)
    segmented_image = segmented_image / np.float32(
        255)  # Note to self: deprecate

    if not unet_3d:
        segmented_image = np.expand_dims(segmented_image, 3)

    if unet_3d:
        segmented_image = np.expand_dims(segmented_image, 4)
        print(segmented_image.shape)

    logging.debug("xray shuffle.")
    np.random.seed(random_seed)
    np.random.shuffle(xray_image)
    np.random.seed(random_seed)
    np.random.shuffle(segmented_image)

    logging.debug("Setting up training and validation images.")
    validation_percentage /= 100
    training_cut = 1 - validation_percentage
    training_cut = round(xray_image.shape[0] * training_cut)
    training_xray = xray_image[:training_cut, :, :]
    validation_xray = xray_image[training_cut:, :, :]
    del xray_image
    gc.collect()

    logging.debug("Setting up training and validation masks.")
    training_mask = segmented_image[:training_cut, :, :]
    validation_mask = segmented_image[training_cut:, :, :]
    del segmented_image
    gc.collect()

    logging.debug("Model training starts.")
    results = ut.fit(training_xray,
                     training_mask,
                     validation_xray,
                     validation_mask,
                     filters=num_of_filters,
                     depth=num_of_layers,
                     patience=training_patience,
                     epochs=training_epochs,
                     model_name=model_name,
                     batch_size=batch_size,
                     unet_3d=unet_3d)
    logging.info(f"Model training completed and saved as: {model_name}.")

    logging.debug("Saving model history data.")
    np.save(model_name + ".npy", results.history)
    logging.info(f"Model history saved as {model_name}.")

    del training_xray, training_mask, validation_xray, validation_mask
    tf.keras.backend.clear_session()


# Training, this part is to be customed by user to fit the problem.

patch_ = 512

print('Setting file directory', flush=True)
parent_dir = "/lustre/or-scratch/cades-birthright/proj-shared/vsv/data/rate_1mL_processed/"
xray_path = "01_not_segmented/"
solid_path = "02_initial_time/"
mask_path = "04_segmented_mineral/"
xray = os.path.join(parent_dir, xray_path)
solid = os.path.join(parent_dir, solid_path)
mask = os.path.join(parent_dir, mask_path)
print('File directory set', flush=True)

print('Loading Images....', flush=True)
xray1 = io.volread(xray + '02_run02_0027min_trsmd.tif')
xray1 = ((xray1 - np.min(xray1)) * np.float32(255.0)) / (np.max(xray1) -
                                                         np.min(xray1))
xray1 = xray1[4:-4, :, :]
xray1 = pt.patch(xray1, 1, patch_, patch_)

xray2 = io.volread(xray + '02_run02_0076min_trsmd.tif')
xray2 = ((xray2 - np.min(xray2)) * np.float32(255.0)) / (np.max(xray2) -
                                                         np.min(xray2))
xray2 = xray2[4:-4, :, :]
xray2 = pt.patch(xray2, 1, patch_, patch_)

xray3 = io.volread(xray + '03_run02_0160min_trsmd.tif')
xray3 = ((xray3 - np.min(xray3)) * np.float32(255.0)) / (np.max(xray3) -
                                                         np.min(xray3))
xray3 = xray3[4:-4, :, :]
xray3 = pt.patch(xray3, 1, patch_, patch_)

xray4 = io.volread(xray + '04_run02_0271min_trsmd.tif')
xray4 = ((xray4 - np.min(xray4)) * np.float32(255.0)) / (np.max(xray4) -
                                                         np.min(xray4))
xray4 = xray4[4:-4, :, :]
xray4 = pt.patch(xray4, 1, patch_, patch_)

xray5 = io.volread(xray + '05_run02_0362min_trsmd.tif')
xray5 = ((xray5 - np.min(xray5)) * np.float32(255.0)) / (np.max(xray5) -
                                                         np.min(xray5))
xray5 = xray5[4:-4, :, :]
xray5 = pt.patch(xray5, 1, patch_, patch_)
print('Images loaded...', flush=True)

print('xray shuffle....', flush=True)
np.random.seed(1)
np.random.shuffle(xray1)
np.random.seed(2)
np.random.shuffle(xray2)
np.random.seed(3)
np.random.shuffle(xray3)
np.random.seed(4)
np.random.shuffle(xray4)
np.random.seed(5)
np.random.shuffle(xray5)
print('xray shuffle complete....', flush=True)

print('Setting up training and validation images....', flush=True)
valid_cut = round(xray1.shape[0] * 0.8)
training_xray = np.concatenate(
    (xray1[:valid_cut, :, :], xray2[:valid_cut, :, :], xray3[:valid_cut, :, :],
     xray4[:valid_cut, :, :], xray5[:valid_cut, :, :]))
xsize = xray1.shape
del xray1, xray2, xray3, xray4, xray5
gc.collect()
print('Training, Validation split of images complete....', flush=True)

print('Loading time 0....', flush=True)
mask1 = io.volread(solid + '02_run02_0027min_trsmd_seg.tif')
mask1 = mask1[4:-4, :, :]
mask1 = pt.patch(mask1, 1, patch_, patch_)
print('Time 0 loaded...', flush=True)

print('Loading Masks....', flush=True)
mask2 = io.volread(mask + '02_run02_0076min_trsmd_subtr_seg.tif')
mask2 = pt.patch(mask2, 1, patch_, patch_)
mask2 = np.invert(mask2) + mask1

mask3 = io.volread(mask + '03_run02_0160min_trsmd_subtr_seg.tif')
mask3 = pt.patch(mask3, 1, patch_, patch_)
mask3 = np.invert(mask3) + mask1

mask4 = io.volread(mask + '04_run02_0271min_trsmd_subtr_seg.tif')
mask4 = pt.patch(mask4, 1, patch_, patch_)
mask4 = np.invert(mask4) + mask1

mask5 = io.volread(mask + '05_run02_0362min_trsmd_subtr_seg.tif')
mask5 = pt.patch(mask5, 1, patch_, patch_)
mask5 = np.invert(mask5) + mask1
print('Masks loaded...', flush=True)

print('xray shuffle....', flush=True)
np.random.seed(1)
np.random.shuffle(mask1)
np.random.seed(2)
np.random.shuffle(mask2)
np.random.seed(3)
np.random.shuffle(mask3)
np.random.seed(4)
np.random.shuffle(mask4)
np.random.seed(5)
np.random.shuffle(mask5)
print('xray shuffle complete....', flush=True)

print('Setting up training and validation mask....', flush=True)
training_mask = np.concatenate(
    (mask1[:valid_cut, :, :], mask2[:valid_cut, :, :], mask3[:valid_cut, :, :],
     mask4[:valid_cut, :, :], mask5[:valid_cut, :, :]))
del mask1, mask2, mask3, mask4, mask5
gc.collect()
print('Training, Validation split of masks complete....', flush=True)

unet_buki_trainer(xray_image=training_xray,
                  segmented_image=training_mask,
                  image_patch=None)
