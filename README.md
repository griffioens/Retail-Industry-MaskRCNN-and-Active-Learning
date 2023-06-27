# MaskRCNN Active Learning
 This projects combines the Mask R-CNN method from the detectron2 package with various active learning methods on a retail dataset.

 The main folder is "iterative active learning process", which  contains the files that perform the iterative process as detailed in algorithm 1 for the Least Confident, Entropy, and CEAL active learning method on two subsets of the dataset.
 
![image](https://github.com/griffioens/Retail-Industry-MaskRCNN-and-Active-Learning/assets/91024075/240aacea-fe4f-4904-a622-17dd284732a0)


The other folders contain support results. The folder BenchMarkResults are the results of the Mask R CNN model trained on the complete dataset. The Hyperparameter optimization contains all the tests conducted to optimize the hyperparameters for the experiment. 

# Installation instructions
1. Install pytorch using the helper tool here: https://pytorch.org/get-started

The code in this project was run on pytorch 1.11, torchvision 0.12 and cudatoolkit 11.3, using the following code.
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

2. Install the altered version of detectron2 using:
python -m pip install -e detectron2


3. Install the latest version of fiftyone for visualization of predictions


# Alteration of the detectron2 package. 
This project uses a slightly altered version of the detectron2 package. The function fast_rcnn_inference_single_image in "detectron2/detectron2/modeling/roi_heads/fast_rcnn.py" is altered to return the confidence score for all decision classes instead of only the most probably class, for the purpose of calculating Entropy.

All rights for the detectron2 package remain with the original owner of the package, found here: https://github.com/facebookresearch/detectron2

# Dataset
The dataset used is the MVTEC D2S Dataset, which can be found here: https://www.mvtec.com/company/research/datasets/mvtec-d2s. All rights for the MVTEC D2S Dataset remain with the original owner. 
