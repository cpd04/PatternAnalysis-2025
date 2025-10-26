# 3D Improved UNet Image Segmentation - Prostate Case Study
This project is the final report assessment task for the University of 
Queensland's COMP3710 - Pattern Recognition and Analysis course. Specifically,
this project implements the [3D Improved UNet](https://arxiv.org/pdf/1802.10508) 
developed by Isensee et al. for the task of segmenting a downsampled version of
the [Prostate 3D MRI data](https://data.csiro.au/collection/csiro:51392v2?redirected=true).

MAKE SURE TO COME BACK AND INCLUDE KEY RESULTS

## Dataset
- Small discussion of the dataset itself before getting into the architecture

## 3D Improved Unet
- How the overall architecture works

### Context Module
- How the context module is developed

### Segmentation Layers
- How the segemntation layers works

### Output Layer
- How the output layer works

### Downsampling
- How the downsampling path works

### Upsampling
- How the upsampling path works

## Usage
- Explanation of the sequence of how to train, predict, test, each of the 
differnt modules etc.

### Dependices
- List all dependices

## Results
- Speak on the overall sequences and final key outcomes produced by the model 

### Training
- Give an overview of the training process

### Performance Tuning
- Give an overview of the training and validation data and how each of the 
hyperparameters were finally selected. Validation data should achieve above
the required accuracy.
- Finally give an example of the performance for the overall test set with
all of the test metrics. 

### Testing
- A summary of key outcomes from the testing dataset

### Visualisation
- Give an example visualisation of at least three different ones from the test
set

## References
Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online] 
Available: https://arxiv.org/pdf/1802.10508

O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Vol-
umetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Inter-
vention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu,
G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432. 
Available: https://arxiv.org/abs/1606.06650 

W. Dai, B. Woo, S. Liu, M. Marques, C. B. Engstrom, P. B. Greer, S. Crozier, J. A. Dowling, and S. S. Chandras,
“CAN3D: Fast 3D Medical Image Segmentation via Compact Context Aggregation,” arXiv:2109.05443 [cs,
eess], Sep. 2021, arXiv: 2109.05443. [Online]. Availabe https://arxiv.org/abs/2109.05443 