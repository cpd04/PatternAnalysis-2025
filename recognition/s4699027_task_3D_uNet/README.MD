# 3D Improved UNet Image Segmentation - Prostate Case Study
This project is the Pattern Analysis task for the University of 
Queensland's COMP3710 - Pattern Recognition and Analysis course. Specifically,
this project implements the [3D Improved UNet](https://arxiv.org/pdf/1802.10508) 
developed by Isensee et al. for the task of segmenting a down-sampled version of
the [Prostate 3D MRI data](https://data.csiro.au/collection/csiro:51392v2?redirected=true).
The project aims to automatically segment the body outline, bone,
bladder, rectum, and prostate with a minimum Dice Similarity Score of 0.7 for 
each class. 

The implementation was a success, being able to achieve a minimum Dice 
Similarity Score of 0.8179 and a maximum of 0.9951 for any individual class. 

## Dataset
The [Prostate 3D MRI dataset](https://data.csiro.au/collection/csiro:51392v2?redirected=true) 
is a 3D dataset comprised of 211 Nifti models from the MRI scan of 38 patients. 
Each scan has also been segmented manually by a MR physicist with more than 
10 years of experience into 6 key categories: background, body outline, bone, 
bladder, rectum, and prostate. 

This project uses these segmented images as the ground truth for the automated 
segmentation task aimed to be achieved through the use of a 3D Improved UNet. 


## 3D Improved UNet
The [3D Improved UNet](https://arxiv.org/pdf/1802.10508) was proposed by Isensee 
et al. in the BRATS 2017 Challenge for identifying different parts of brain 
tumours. The architecture was inspired by the original 
[3D UNet](https://arxiv.org/abs/1606.06650) developed by Cicek et al., which 
itself was a derivative of the original 
[2D UNet](https://arxiv.org/abs/1505.04597) proposed for medical imaging. 

![Architecture](./assets/Improved_3D_UNet_Architecture.png)
*Reference: [Isensee et al.](https://arxiv.org/pdf/1802.10508)*

### UNet Comparison
The 3D Improved UNet improves upon the original 3D UNet in a range of ways. 

1. Deeper than the original 3D Unet (5 layers instead of the 4 layers).
2. Utilises Leaky ReLU over the traditional ReLU with a negative slope of 0.01.
3. Uses instance normalisation over batch normalisation as the stochasticity induced by the small batch size may destabilise batch normalisation.
4. Introduces residual  connections within the aggregation and context pathway.
5. Loss function uses soft dice loss only instead of the original weighted cross-entropy loss.
6. Uses upsampling modules instead of transposed up-convolution to avoid checker-board artifacts.
7. Segmentation modules at used in layers 3, 2, and 1 and later combined to achieve a higher accuracy.

### Modules
- **Context Module** - A context module aims to achieve information from the broader context, and is simply a pre-activation residual block. There are two 3x3x3 convolutions layers separated by a dropout layer of 0.3.
- **Upsampling Module** - An upsampling module takes in low-resolution feature maps and upscales the compressed information. It is comprised of a simple upsample that doubles the features in each of the three spatial dimensions, before apply a 3x3x3 convlution to halve the number of features.
- **Localisation Module** - A localisation module recombines the higher dimensionality feature map with the outputs from the context modules after they have been convolved together, reducing the number of feature maps for better memory consumption. It is comprised of a 3x3x3 convolution followed by a 1x1x1 convolution to halve the number of feature maps.
- **Segmentation Layer** - A segmentation layer is used to get segmentation information at different levels in the UNet architecture for better performance. It uses a 1x1x1 convolution with the number of output channels being the number of labels in the model to be segmented.

### Aggregation and Context Pathway
The aggregation pathway aims to gain broader contextual information from the 
entire image. It begins with a 3x3x3 convolution followed by a context module. 
The output of the context module and the convolution are then summed together. 
Following the initial layer, the same process is repeated four more times. 
However, instead a stride of 2 is used in the initial convolution instead of 1.

### Localisation Pathway
The localisation pathway utilises the developed upsampling, localisation, and 
segmentation modules to localise the information gained and segment the model. 
On layer 5, the beginning of the localisation pathway, a single upsampling 
module is used. The output of this upsampling module is then concatenated with 
the output of the context module of the layer above. The output of this 
concatenation is then fed into a localisation module, and the process repeats 
for layers 4, 3, and 2. On layers 3 and 2 a segmentation layer is added, taking 
in the output of the localisation module and segmenting the image. The 
segmentation from layer 3 is simply 2x upscaled in each of the spatial direction 
before being added to the segmentation from layer 2. Layer 1 is unique in which 
it contains only a 3x3x3 convolution with a stride of 1 after the concatenation 
of the context module with the localised information from the layer below. The 
output of this convolution is then segmented in a segmentation layer, and summed 
with the combined output of Layer 3 and 2, which has again been upsampled, to 
reach the correct dimension. Finally, a softmax activation function is 
implemented to give class probabilities.

## Usage
To reproduce the predictive model ensure the following:
1. Clone this repository into the working directory
2. Install the necessary dependencies 
3. [Download the data](https://data.csiro.au/collection/csiro:51392v2?redirected=true) and create a new folder called 'data' in the same folder as train.py. Copy the downloaded data into this data folder. The structure should appear as follows:

<pre>
> PatternAnalysis-2025
__> recognition
____> s4699027_task_3D_uNet
______> assets
______> data                                  <-- Copy data into data folder
________> HipMRI_study_complete_release_v1    <-- Downloaded data
______> .gitignore
______> dataset.py
______> modules.py
______> predict.py
______> README.MD
______> train.py
______> utils.py
</pre>

To train, validate, and test the 3D Improved UNet model on the dataset, with visualisations at each epoch, run the following command:
<pre>
python train.py
</pre>

To use the saved model, if it exists, to predict on the test set, evaluate the model, and visualise the outputs, run the following command:
<pre>
python predict.py
</pre>

Both train.py and predict.py will automatically generate a model folder, a 
visualisation folder, and a wandb folder in the same folder where train.py is 
present. If the desired location for these outputs is not in the same folder, 
they can be changed by changing the ```MODEL_PATH```, ```VISUAL_PATH_INPUTS```, 
```VISUAL_PATH_VALIDATION```, and ```VISUAL_PATH_OUTPUTS``` global parameters 
under train.py.

### Dependices
The following dependencies are necessary:
<pre>imageio 2.37.0
matplotlib 3.10.7
nibabel 5.3.2
numpy 2.2.6
torch 2.9.0
torchio 0.21.0
torchvision 0.24.0
tqdm 4.67.1
wandb 0.22.3
</pre>

The [wandb library](https://wandb.ai/site/) allows users to keep track of the 
accuracy and loss of the model during and after training. An account is required 
to use this library, and at the start of each session an API key from the 
account is required. Refer to the Weights & Biases website for further details.

## Results
### Training
The training process requires the use of an a100 GPU due to the high VRAM 
requirements of the large convolution processes. The model was train with the 
following hyperparameters:
- **Batch Size**: 2
- **Initial Learning Rate**: 4e-4
- **Final Learning Rate**: 1e-4
- **Epochs**: 30
- **Scheduler**: Cosine Annealing Learning Rate Scheduler
- **Optimiser**: adam

A common issue with training for medical imaging is overfitting to a certain 
representation of data. For example, the segmentation may only work in a 
particular orientation. One way to overcome this is with data pre-processing and 
augmentation to the training and validation set. The following transforms were 
applied:
- **Horizontal Flipping**: Used to ensure orientation does not matter, 50% of images.
- **Vertical Flipping**: Used to ensure orientation does not matter, 50% of images.
- **Noise**: Noise is a common issue with medical images, and as such an additional gaussian noise was added, 50% of images.
- **Blur**: Forces model to learn more robust features and ignore sharp details, 25% of images.
- **Gamma**: Adjusts image intensity with non-linear gamma correction, ensures different insenity distrbutions do not affect outcomes. 25% of images.

Finally, as per the model architecture, a multiclass soft dice loss was used as 
the criterion of evaluation. Soft dice loss measures the overlap between two 
sets, which is better for evaluating segmentation performance and much better 
for class imbalances commonly found in medical imaging. As it is a loss 
function, the more negative the higher accuracy the result.

Interestingly, it appears that the initialisation of the weights has a large 
impact on the training accuracy. Well-initialised results are able to achieve
a high dice similarity score within very few epochs, while poorly-initialised
results often mask correctly but assign the mask the incorrect label. It is
assumed with an increase in the number of epochs or a re-initialisation of the
model the high accuracy presented can be achieved.

During the training, the following training/validation curve for the soft dice 
loss was produced. The closeness in both trianing and validation losses indicate 
that no overfitting to the training data was present. 

![Training_Accuracy_Sequence](./assets/training_vs_val_loss.png)

The average multiclass dice coefficient and the validation dice similarrity 
coefficients for each indiivdual class were also plotted. The results show 
clearly the minimum accuracy can be achieved for the transformed validation 
dataset.

![Average_Multiclass_Dice_Coefficient](./assets/multiclass_dice_similarity_score.png)

![Single_Dice_Coefficient](./assets/dice_similarity_score.png)

### Visualisation over Epochs
A set of visualisations over each epoch was also generated to see how 
improvements over time occur. The input can be seen on the left, the true 
segmented image in the middle, and the prediction on the right.

| Epoch | Visualisation                                | Validation Loss  | Training Loss  | Multiclass Dice Score  |
|-------|----------------------------------------------|------------------|----------------|------------------------|
| 1     | ![Val_Epoch_1](./assets/val_epoch_1.gif)     | -0.5345          | -0.3930        | 0.5446                 |
| 3     | ![Val_Epoch_3](./assets/val_epoch_3.gif)     | -0.6557          | -0.6582        | 0.6711                 |
| 5     | ![Val_Epoch_5](./assets/val_epoch_5.gif)     | -0.7861          | -0.7467        | 0.7917                 |
| 10    | ![Val_Epoch_10](./assets/val_epoch_10.gif)   | -0.8577          | -0.8599        | 0.8638                 |
| 20    | ![Val_Epoch_20](./assets/val_epoch_20.gif)   | -0.9131          | -0.8943        | 0.9021                 |
| 30    | ![Val_Epoch_30](./assets/val_epoch_30.gif)   | -0.9319          | -0.8998        | 0.9046                 |


### Testing
Having successfully trained the model, key evaluations were made against the 
test set. The minimum Dice Similarity Score for each class independently was 
required to be above 0.7, which was easily passable by the developed model. 

The Dice Similarity Score for each class was evaluated as follows:
| Class | Structure      | Score   |
|-------|----------------|---------|
| 0     | Background     | 0.9951  |
| 1     | Body Outline   | 0.9795  |
| 2     | Bone           | 0.8855  |
| 3     | Bladder        | 0.9149  |
| 4     | Rectum         | 0.8179  |
| 5     | Prostate       | 0.8353  |

The overall test dice score for all of the classes was also 0.9047, again 
confirming the high accuracy of the model.

A sample visualisation on the test set of the segmented image of this model can 
be seen below. The input can be seen on the left, the true segmented image in 
the middle, and the prediction on the right. Note that the classes are in order 
of darkest to lightest when visualised.

![Test Set Image](./assets/test_set_1.gif)  


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

O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241. 