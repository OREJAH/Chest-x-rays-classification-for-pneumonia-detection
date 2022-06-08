# Chest x-rays classification for pneumonia detection

# Table of Contents
## * Review of the Project

## * Summary
   ### ~ Statement of the problem
   ### ~ Solution
     
## * Project Environment Set Up

## * Dataset  
   ###  ~ Access
   ###  ~ Overview and Structure
   ###  ~ Task

## * Dataset Collection and Processing

## * Image Data Loading and Transformation
   
## * Building ResNet50 Model

## * Model Training

## * Model Evaluation on Test Dataset

## *  Recommendations
<br>
<br/>
<br> <br/>

## * Review of the Project
In this project, a binary classification model will be built using chest x-ray images. This classification system is targeted at children, adults over the age of 65 and other individuals with underlying respiratory illness. There is a scarce availability of healthcare practitioners and radiologists around the globe whose prediction on pneumonia cases matter greatly. Therefore, this deep learning model using Convolutional Neural Network (CNN) would give an higher accuracy in the prediction of pneumonia and it will further stand as a prelude to serve as a better model for other life-threatening illnesses.

## * Summary
   ### ~ Statement of the problem
Pneumonia is a contagious disease that has been observed to be the biggest cause of child death in Nigeria, killing more than 162,000 children under the age of five in 2018, of more than 18 children per hour.
There is a lack of intensive longitudinal monitoring systems that helps in measuring the impact of preventive measures in children by addressing underlying causes of pneumonia. Similarly, there is a lack of strong and equitable health systems to adequately prevent, diagnose and treat pneumonia, and provide children with their basic human right to good-quality healthcare.

   ### ~ Solution
As a result of the aforementioned observations, there is a need to build a classification model for pneumonia detection.

## * Project Environment Set Up
Google Colaboratory is a free online cloud-based Jupyter notebook environment where machine learning and deep learning models can be trained on CPUs, GPUs, and TPUs. Since the models that will be utilized in this project are large and may take a long time to execute on a standard CPU, I'll be making use of Colab's CUDA-enabled GPU to help the models get trained faster.

### Setup
. Create a Google CoLab notebook: [This link](https://colab.research.google.com/) will take you to Google CoLab. To make a new CoLab notebook, click the NEW NOTEBOOK icon. By pressing the upload option, you may also upload your local notebook to CoLab. You may rename your notebook by simply clicking on it and changing the name to whatever you like. I normally name them after the project that I'm working on.

. Access the GPU on the CoLab notebook: On the top menu, click 'Runtime' and then 'Change Runtime Type'. You should see a drop-down with hardware accelerators that CoLab provides, click on GPU and click on SAVE.

. Connecting the GPU: On the top right of the notebook, click on the 'connect' option as this will connect to a virtual machine with GPU. Once it is connected, you'll then be able to see the amount of RAM and storage available to execute our python codes.

## * Dataset

 ###  ~ Access
 The dataset used for this project is gotten from Kaggle's repository and can be found in this [link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

###  ~ Overview and Structure
This dataset comprises chest x-ray images, some of which contain chest x-rays that do not show pneumonia. We have three types of chest x-rays present in the dataset: normal chest x-rays, bacterial chest x-rays, and viral pneumonia chest x-rays. 
There are no white clouds on normal chest x-rays. Bacterial pneumonia chest x-rays have focal white clouds while viral pneumonia chest x-rays shows more widespread white clouds.
![x-rays](https://github.com/OREJAH/Chest-x-rays-classification-for-pneumonia-detection/blob/main/x-rays.PNG)
 ###  ~ Task
 The goal of this project is to identify the chest x-rays that are affected by pneumonia.
 
## * Dataset Collection and Processing
### Collection
There are several methods for importing data into jupyter notebooks. For this project, the Kaggle dataset was downloaded and uploaded to Dropbox, a cloud storage service. The dataset may be downloaded into the notebook with only one line of code.

**wget** is used to access files through http or https. The exclamation mark **!** is a command that tells the Linux terminal to run this line. The dataset is then downloaded as a zip file. To unzip the zip file, we use the !unzip  -q  "archive.zip"

To get the project off to a good start, all of the libraries needed for the completion of this project will be imported. They are:

     > import tensorflow as tf 
     
     > import numpy as np
     
     > import matplotlib.pyplot as plt
     
     > from pathlib import path
Then, the global random seed is set to 4.
     
     > tf.random.set_seed(4)
     
### Processing
Creation of pathlib path objects for the train, test and validation directories.

     > train_path = path("chest_xray/train/")
     > validation_path =path("chest_xray/val")
     > test_path = path("chest_xray/test")
     
After creating the pathlib data, glob method is then used.  
All the paths inside the noraml and pneumonia folders are therefore collected using glob.

     > train_image_paths = train.path.glob("*/*")
     > val_image_paths = validation_path.glob("*/*")

Pathlib's glob method returns a generator objectwhen the training image path are printed:

     >  print(train_image_paths)
     
## * Image Data Loading and Transformation

Various libraries exist to load picture paths as a vector of pixels. Tensorflow's most recent version now allows you to load pictures directly. In this section, the load_and_transform function is created and takes in several arguments.

The **TENSORFLOW.IO.READ_FILE** function accepts a file path as a string input and returns an encoded string of that file once it has been read. Because the images in this project are in JPEG format, we utilize the function TENSORFLOW.IO.DECODE_JPEG.

The function **TENSORFLOW.IMAGE.RESIZE** resizes the tensor using two arguments:
 - *images* which is the tensor returned by the decode function
 - *size* which is the size to which it must be resized, since ResNet50 is being used for this project, the images have been resized to 224.
It further takes an argument named *method*, which specifies the techniques used to interpolate the pixels whether they are compressed or expanded.

Next, a data augmentation function (**TENSORFLOW.IMAGE.RANDOM_FLIP_LEFT_RIGHT**) is used as it helps to replace the current picture with an alternate version of the same image. Because the model is searching in seven distinct regions, it is able to generalize better over photos.
Finally, we restore the tensor to its original label.

## * Building ResNet50 Model

After the **Get_dataset** function is constructed, and it accepts three arguments: image paths, labels, and the state of processing, which is either training or not, then, we use the *convert_to_tensor* method, thsi helps the list of paths and labels to be turned into tensorflow tensors. Next, for each path and label, we build a dataset object using *from_tensor_slices* method. Then zip the two dataset objects and apply random shuffle.

*Image To build the model, 

    




## * Model Training
*Image



## * Model Evaluation on Test Dataset
In preparation of testing the model on the dataset, the steps done during model training are also repeated, starting from collecting all data paths using pathlib object to convert all the paths to tensorflow tensors. The evaluate method returns the metrics (accuracy, precision and recall).



## *  Recommendations
Trying more augmentations is a risky component of medical imaging, but it's a good way to improve the model. Selecting a safe augmentation that does not change the pixel values might be beneficial. More *callbacks* are effective in monitoring metrics and preventing overfitting of the model being trained.
 
## License

See the [LICENSE](https://github.com/OREJAH/Chest-x-rays-classification-for-pneumonia-detection/blob/main/LICENSE.md) file for license rights and limitations.
