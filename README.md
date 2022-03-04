# Chest-x-rays-classification-for-pneumonia-detection

# Table of Contents

## * Review of the Project

## * Project Environment Set Up

## * Dataset
   ###  ~ Overview
   ###  ~ Task
   ###  ~ Access

## * Dataset Collection and Processing

## * Image Data Loading and Transformation

## * Tensorflow Dataset Object Creation

## * Convolutional Neural Network (ResNets)
   
## * Building ResNet50 Model

## * Model Training

## * Results Interpretation

## * Model Evaluation on Test Dataset

## * Future Work
<br>
<br/>
<br> <br/>

## * Review of the Project

In this project, a binary classification model will be built using chest x-ray images. This classification system is targeted at children, adults over the age of 65 and other individuals with underlying respiratory illness. There is a scarce availability of healthcare practitioners and radiologists around the globe whose prediction on pneumonia cases matter greatly. Therefore, this deep learning model using Convolutional Neural Network (CNN) would give an higher accuracy in the prediction of pneumonia and it will further stand as a prelude to serve as a better model for other life-threatening illnesses.

## * Project Environment Set Up
Google Colaboratory is a free online cloud-based Jupyter notebook environment where machine learning and deep learning models can be trained on CPUs, GPUs, and TPUs. Since the models that will be utilized in this project are large and may take a long time to execute on a standard CPU, I'll be making use of Colab's CUDA-enabled GPU to help the models get trained faster.

### Setup
. Create a Google CoLab notebook: [This link](https://colab.research.google.com/) will take you to Google CoLab. To make a new CoLab notebook, click the NEW NOTEBOOK icon. By pressing the upload option, you may also upload your local notebook to CoLab. You may rename your notebook by simply clicking on it and changing the name to whatever you like. I normally name them after the project that I'm working on.

. Access the GPU on the CoLab notebook: On the top menu, click 'Runtime' and then 'Change Runtime Type'. You should see a drop-down with hardware accelerators that CoLab provides, click on GPU and click on SAVE.

. Connecting the GPU: On the top right of the notebook, click on the 'connect' option as this will connect to a virtual machine with GPU. Once it is connected, you'll then be able to see the amount of RAM and storage available to execute our python codes.

## Dataset

## License

See the [LICENSE](https://github.com/OREJAH/Chest-x-rays-classification-for-pneumonia-detection/blob/main/LICENSE.md) file for license rights and limitations.
