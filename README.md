# CigRockSEM Dataset

The **CigRockSEM** dataset is a collection of rock scanning electron microscope (SEM) images, designed for image segmentation tasks. This repository contains code for pre-processing the raw SEM images and implementing traditional and deep learning-based image segmentation methods. Additionally, it includes performance evaluation metrics to assess the effectiveness of different segmentation techniques.

## Project Structure

- **Pytorch-UNet-master**: This folder contains the implementation of the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) repository. It provides the U-Net architecture for deep learning-based segmentation of rock SEM images.
- **pytorch-deeplab-xception-master**: This folder contains the implementation of the [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) repository. It includes the DeepLab v3+ model with Xception backbone for semantic segmentation tasks.
- **Image Preprocessing**: Scripts for preprocessing SEM images, including resolution adjustment, median blurring, CLAHE, cropping:
  - `mudstone_UniformResolution_upsample.py`: Upsamples mudstone SEM images to a uniform resolution.
  - `shale_UniformResolution_upsample.py`: Upsamples shale SEM images to a uniform resolution.
  - `MedianBlur.py`: Applies a median blur filter to the images.
  - `CLAHE.py`: Applies CLAHE to the images.
  - `crop_512_512.py`: Crops images to a standard 512x512 resolution.
  - `Oust_method.py`: Performs Oust method for segmentation.
  - `adaptive_thresh.py`: Performs adaptive thresholding for segmentation.
  - `water.py`: Contains scripts for handling water segmentation.
  - `kmeans_segmentation.py`: Implements the k-means clustering algorithm for image segmentation.
  - `SegEvalMetrics.py`: Contains metrics for evaluating segmentation performance.

## Acknowledgements
The deep learning models used in this project are based on the following repositories:
- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) by Milesial for U-Net implementation.
- [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) by jfzhang95 for DeepLab v3+ Xception backbone implementation.

These models have been adapted for rock SEM image segmentation tasks.
