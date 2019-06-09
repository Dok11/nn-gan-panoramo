# nn-gan-panoramo
Implementation of Generative Adversarial Network for equirectangular panorama generation


# Project structure

`/data/train*.npz` - File with Numpy array of images with 128x64x1 dimension  
`/images` - Presentation of images with training result at several epochs  
`/train-images` - Dataset with panorama images (jpg) with 640x320x3  
`/gan.py` - GAN based on dense layers for generation panorama images 128x64x1  
`/gan_src.py` - GAN based on dense layers for MNIST digits generation 28x28x1. Copy from https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/master/gan/gan.py  
`/prepare-data.py` - Script for creation npz file with images from `/images`  
