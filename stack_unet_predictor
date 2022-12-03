author__ = "Olubukola Ishola"

__email__ = "olubukoa.ishola@okstate.edu"
"""
@author: Olubukola Ishola
"""

import patchify_buki as pt
import imageio as io
import numpy as np
import unet3D_buki_algo as ut
import tensorflow as tf
from scipy.ndimage import measurements
import logging

logger = logging.getLogger(__name__)


def unet_buki_predictor(xray_image,
                        model,
                        image_patch=512,
                        model_type='2D',
                        threshold=128,
                        unet_3d=False):
    """
    This is just function to predict from a segmenatation model using the unet_buki_algo and patchify_buki.

    Parameters
    ----------
    xray_image : array
        Xray image to be segmented.
    model : '.h5', optional
        Saved unet model. The default is 'model2D_solid_filters_16_depth_4_img_patch_256_grain_shuffle'.
    img_patch : int, optional
        The height and width of the image. Note, the height and width must be equal. Note, this must be guided by the dimensioons used to train your model. The default is 256.
    model_type : int, optional
        Specify if the model is 2D or 3D. For 3D unet, the third dimensionis equal to img_patch. The default is '2D'.
    threshold : int, optional
        The cut oof point for the binary task,  it must be a number between 0 and 255. The default is 128.
    unet_3d : boolean, optional
        Activates 3D unet inplace of 2D unet. The default is False.
        
    Returns
    -------
    segmented_image : array of uint8. 
        Segmented image. Note: this funnction will not export the image. You need to write a line of code to write the image to file.

    """

    logging.debug(
        "Setting up data for loaded model.")  # Note to self: deprecate
    if model_type == '2D':
        img_depth = 1
    elif model_type == '3D':
        img_depth = image_patch
    else:
        raise ValueError("Specify model_type as 2D or 3D.")

    tf.keras.backend.clear_session()

    logging.debug("Loading Image.")
    xray_image = ((xray_image - np.min(xray_image)) *
                  np.float32(255.0)) / np.max(xray_image) - np.min(xray_image)

    logging.debug("Patching image data.")
    dim = [xray_image.shape[0], xray_image.shape[1],
           xray_image.shape[2]]  # Dimension of original images before patching
    xray_image = pt.patch(xray_image, img_depth, image_patch, image_patch)

    logging.debug("Normalizing image data.")
    xray_image = xray_image / np.float32(255)

    logging.debug("Adding channel dimension.")
    if model_type == '2D':
        xray_image = np.expand_dims(xray_image, 3)

    if model_type == '3D':
        xray_image = np.expand_dims(xray_image, 4)

    logging.debug("Predicting mask.")
    segmented_image = ut.predict(xray_image, model, unet_3d=unet_3d)
    del xray_image

    logging.debug("Processing predicted image.")
    segmented_image = (segmented_image > threshold) * 255
    segmented_image = segmented_image.astype(np.uint8)

    logging.debug("Unpatching images.")
    segmented_image = pt.unpatch(segmented_image, dim)

    logging.debug("Segmentation complete.")
    return segmented_image


def cylinder_gen(yCen, xCen, yPoi, xPoi, rad, depth):
    '''
    Description:
        Creates an aray representation of a cylinder.
            
    Parameters: 
        yCen: column location of thecenretr of a circle slice through the cylinder
        xCen: row location of thecenretr of a circle slice through the cylinder
        yPoi: number of columns of the background array
        xPoi: number of rows of the background array
        rad; radius of the cylinder
        depth: height of the cylinder
        
    Returns:
        An aray representation of a cylinder. Cylinder is 0, exterior of cylinder is 255.
    '''

    logging.debug("Creating geometry.")
    print('', flush=True)
    data = np.full((yPoi, xPoi), 255)
    data = data.astype(np.uint8)
    for y in range(yPoi):
        for x in range(xPoi):
            if ((x - xCen)**2 + (y - yCen)**2 - rad**2) <= 0:
                data[y, x] = 0

    data = np.repeat(data[np.newaxis, :, :], depth, axis=0)
    print('done.', flush=True)

    return data


def post_process_trimmer(xray,
                         model='solid_model_patch_512_0',
                         threshold=0,
                         image_patch=512):

    logging.debug("Trimming region of in image."
                  )  # The  choice of model may require tunning for data
    logging.debug("Predicting 0 threshold mask.")
    raw_mask = unet_buki_predictor(xray_image=xray,
                                   threshold=threshold,
                                   image_patch=image_patch,
                                   model=model)

    logging.debug("Box crop around region of interest.")
    sample = raw_mask[0, :, :]
    top, bottom = 0, sample.shape[0] - 1
    left, right = 0, sample.shape[1] - 1

    while sum(sample[top, :]) / (right - left + 1) == 255:
        top += 1

    while sum(sample[bottom, :]) / (right - left + 1) == 255:
        bottom -= 1

    sample = sample[top:bottom, :]

    while sum(sample[:, left]) / (bottom - top) == 255:
        left += 1

    while sum(sample[:, right]) / (bottom - top) == 255:
        right -= 1

    sample = sample[:, left:right]
    bottom += (1316 - sample.shape[0])
    right += (1319 - sample.shape[1])
    sample = raw_mask[:, top:bottom, left:right]

    logging.debug("Cylindrical crop around region of interest.")
    mid_row = round(np.median([0, sample.shape[1]]))
    mid_col = round(np.median([0, sample.shape[2]]))
    rad = (mid_row + mid_col) / 2
    trimmer = cylinder_gen(mid_row, mid_col, sample.shape[1], sample.shape[2],
                           rad, sample.shape[0])

    return trimmer, top, bottom, left, right


def barite_predictor(xray_0, threshold=128):

    mask_0_barite = unet_buki_predictor(xray_image=xray_0,
                                        threshold=threshold,
                                        image_patch=512,
                                        model='barite_model')

    mask_0_barite = mask_0_barite.astype(np.float32)
    mask_0_barite = (mask_0_barite == 255) * 255

    return mask_0_barite.astype(np.uint8)


def stack_unet(xray_0, threshold=128):

    mask_0_16 = unet_buki_predictor(xray_image=xray_0,
                                    threshold=threshold,
                                    image_patch=16,
                                    model='solid_model_patch_16')

    mask_0_32 = unet_buki_predictor(xray_image=xray_0,
                                    threshold=threshold,
                                    image_patch=32,
                                    model='solid_model_patch_32')

    mask_0_64 = unet_buki_predictor(xray_image=xray_0,
                                    threshold=threshold,
                                    image_patch=64,
                                    model='solid_model_patch_64')

    mask_0_128 = unet_buki_predictor(xray_image=xray_0,
                                     threshold=threshold,
                                     image_patch=128,
                                     model='solid_model_patch_128')

    mask_0_256 = unet_buki_predictor(xray_image=xray_0,
                                     threshold=threshold,
                                     image_patch=256,
                                     model='solid_model_patch_256')

    mask_0 = unet_buki_predictor(xray_image=xray_0,
                                 threshold=threshold,
                                 image_patch=512,
                                 model='solid_model_patch_512_0')

    xray_1 = np.rot90(xray_0, k=1, axes=(0, 1))
    mask_1 = unet_buki_predictor(xray_image=xray_1,
                                 threshold=threshold,
                                 image_patch=512,
                                 model='solid_model_patch_512_1')
    mask_1 = np.rot90(mask_1, k=1, axes=(1, 0))

    xray_2 = np.rot90(xray_0, k=1, axes=(0, 2))
    mask_2 = unet_buki_predictor(xray_image=xray_2,
                                 threshold=threshold,
                                 image_patch=512,
                                 model='solid_model_patch_512_2')
    mask_2 = np.rot90(mask_2, k=1, axes=(2, 0))

    logging.debug("Selecting regions where prediction overlaps in all models.")
    mask = (mask_0.astype(np.float32) + mask_1.astype(np.float32) +
            mask_2.astype(np.float32) + mask_0_16.astype(np.float32) +
            mask_0_256.astype(np.float32) + mask_0_64.astype(np.float32) +
            mask_0_128.astype(np.float32) + mask_0_32.astype(np.float32)) // 8
    mask = (mask == 255) * 255

    return mask.astype(np.uint8)


def threshold(mask, threshold=128):  # Note to self: deprecate

    mask = io.volread(mask)
    mask = (mask > threshold) * 255
    mask = mask.astype(np.float32)

    return mask


def post_stack_unet(mask_list):  # Note to self: deprecate

    mask = sum(mask_list) // len(mask_list)
    mask = (mask == 255) * 255

    return mask.astype(np.uint8)


def filler(mask):

    logging.debug(
        "Post segmentation step to remove holes in the solid portion.")

    mask = np.invert(mask)

    logging.debug("Labelling clusters and keeps only the largest cluster.")
    lw, num = measurements.label(mask)
    minLab = np.min(lw)
    maxLab = np.max(lw)
    hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
    maxClLab = np.argmax(hist) + 1
    mask[lw != maxClLab] = 0
    mask = np.invert(mask)
    mask = mask.astype(np.uint8)

    return mask


# 1ml Implemetation starts here

print('Adding input and output folders', flush=True)
inp_loc, out_loc = '/lustre/or-scratch/cades-birthright/proj-shared/vsv/data/rate_1mL_not_processed_many/01_not_segmented/', '/lustre/or-scratch/cades-birthright/proj-shared/oi6/new_processed_1mL/'
# inp_loc, out_loc = 'C:/Users/oishola/OneDrive - Oklahoma A and M System/VirtualBox_Shared/multi_comp_unet/', 'C:/Users/oishola/OneDrive - Oklahoma A and M System/VirtualBox_Shared/multi_comp_unet/__'

print('Adding filenames', flush=True)
filenames = [
    '02_run02_2.3SI_Ba_70-110um_1mL_A__27min_corTlt_crp.tif',
    '05_run02_2.3SI_Ba_70-110um_1mL_J__362min_corTlt_crp.tif'
]

for filename in filenames:
    print('Solid: Reading input micro-CT', flush=True)
    xray_0 = io.volread(inp_loc + filename)

    print('Creating trimmer', flush=True)
    trimmer, top, bottom, left, right = post_process_trimmer(xray_0)

    print('Implememnting stack_unet', flush=True)
    mask = stack_unet(xray_0, threshold=128)

    print('Trimming ensemble mask', flush=True)
    mask = mask[:, top:bottom, left:right]

    print('Cleaining areas out of ROI', flush=True)
    mask = mask.astype(np.float32) + trimmer.astype(np.float32)
    mask[mask >= 255] = 255
    mask = mask.astype(np.uint8)
    print('Removing isolated pore spaces', flush=True)
    mask = filler(mask)
    print('Saving post processed image', flush=True)
    io.volsave(out_loc + 'ensemble_xyz_solid_' + filename, mask)
    print('saved: ' + filename)

    print('Barite prediction', flush=True)

    print('Implememnting stack_unet', flush=True)
    mask = barite_predictor(xray_0, threshold=128)

    print('Trimming ensemble mask', flush=True)
    mask = mask[:, top:bottom, left:right]

    print('Saving post processed image', flush=True)
    io.volsave(out_loc + 'barite_' + filename, mask)
    print('saved: ' + filename)
