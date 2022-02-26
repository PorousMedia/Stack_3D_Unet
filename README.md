Created on Fri Feb 25 09:41:15 2022

@author: 
    Buki: olubukola.ishola@okstate.edu
    Data and Computational Geoscientist

about:
    unet_BukiMod is a modification to the Unet architecture for semantic segmentation of images. 
    This is a useful tool in semantic segmentation of images.
    The modifications are:
        The ability to run Unet as a function without having to code the algorithm
        The ability to reduce the depth (contraction and corresponding expansion) of the Unet algorithm. A standrad Unet has a depth of 4, this modification allows varying depth           of one to ten.
        The ability to adjust the number of filters in the first convolution layer. The default Unet has 16 filters. Lie Unet the number doubles along the contraction path and             halfs along the expansion path.
    Note: 
        This code is developed for grayscale images so the number of channels is one. 
credit: 
    U-net: Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image     computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    DigitalSreeni: https://www.youtube.com/c/DigitalSreeni/
"""
"""
code requirement(s):
    install tensorflow:  "pip install tensorflow" see more information here: https://www.tensorflow.org/
    install keras:  "pip install keras" see more information here: https://pypi.org/project/keras/
"""
