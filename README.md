# CNN-Image-Denoising
Image Denoising with Convolutional Neural Networks

In this project I will follow the architecture described in https://arxiv.org/abs/1608.03981

Specific Goals:
1. To build the architecture proposed in Figure 1. of the above link.

2. Repeat the experiment described in Figure 2. of the above link.

Specifically:

a. Use the Berkeley Segmentation Dataset.

b. Downsample the images by a factor of 2 (to reduce JPEG artifacts). We use “bicubic” interpolation kernel for resizing.

c. Add Gaussian noise with zero mean. Sigma of the noise should be 25 out of 256 grey scale values.

d. Extract 50 x 50 patches from the clean and noisy images of training set.

e. Plot the results.
