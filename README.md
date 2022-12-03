Created on Fri Feb 25 09:41:15 2022

**_Author_**: 
    Buki: olubukola.ishola@okstate.edu
    Machine learning and Computational Geoscientist

**_About_**:
    Stack_3D_UNET is a modification to the UNET architecture for semantic segmentation of images. 
    This is a useful tool in semantic segmentation of images with higher degree of accuracy and at the fraction of computational cost of traditional UNET.
    
**_The key modifications are:_**
    The ability to run Unet as a function without having to code the algorithm.
    The ability to reduce the depth (contraction and corresponding expansion) of the unet algorithm. A standard Unet has a depth of 4, this modification allows varying depth from one to infinity.
    The ability to adjust the number of filters in the first convolution layer. The standard unet has 64 filters. Note that the number doubles along the contraction path and is divide by two along the expansion path.

**_Key improvement on traditional UNET are:_**
    Lower RAM required for training.
    Faster computational time.
    Imporved performance on test data.

**_Note_**: 
    This code is developed for grayscale images so the number of channels is one. RGB to be supported soon. 
    This code is developed for binary tasks. Multiclass tasks to be supported soon. 
    The sample predictor codes can be used along with the models linked to it. It predicts both the solid and barite component of input micro-CT image. If it doesn't work for you, you need to train with your own data from scratch. Transfer learning to be supported soon.

**_Credit_**: 
    U-net: Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    DigitalSreeni: https://www.youtube.com/c/DigitalSreeni/

**_Code requirement(s)_**:
    install tensorflow:  "pip install tensorflow" see more information here: https://www.tensorflow.org/
    install keras:  "pip install keras" see more information here: https://pypi.org/project/keras/
