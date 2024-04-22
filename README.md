# Multi-label-Image-Classification
Final project - CS 596 – Special Topics on Deep Learning

The aim of this final project is to implement a multi-label image classifier on the [PASCAL VOC 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). We will design and train deep convolutional networks to predict a binary present/absent image-level label for each of the 20 PASCAL classes.

This will help us gain experience with PyTorch. We will do the following: 

• Part A - Pre-defined Models: Warm-up practice with PyTorch. Run three experiments (listed below) and report your results and observations. MP3_P1A_Introduction.ipynb will guide you to start.<br>
➢ Train AlexNet (PyTorch built-in) from scratch.<br>
➢ Fine-tune AlexNet (PyTorch built-in), which is pretrained on ImageNet.<br>
➢ Train a simple network (defined in classifier.py) from scratch.<br>

• Part B - Self-designed Models: Design your own model architecture to solve this multi-label classification task. MP3_P1B_Develop_Classifier.ipynb will guide you to start.<br>
➢ No pre-trained model is allowed for Part B.<br>
➢ You can use the concepts or ideas from existing models (e.g. VGG, ResNet, DenseNet, etc.), but please don't just copy the existing models. We want to see your own innovation reflected in the model design.<br>
➢ You may want to start your design from either a simple network provided in classifiers.py or AlexNet.<br>

### Data Setup (Local)
Once you have downloaded the zip file, create a folder Project and execute the
download_data script provided:

    /download_data.sh

### Data Setup (For Colaboratory)
If you are using Google Colaboratory for this assignment you will need do some additional setup steps. 

You will need to run the download_data.sh script inside colab cells and move all the data to a local directory in colab for fast access.

You will now need to open the assignment 3 ipython notebook file from your Google Drive folder in Colaboratory and run a few setup commands. Make sure to set the GPU3 as the hardware accelerator. To do this, on the top bar choose `Edit->Notebook Settings->
select 'GPU'`.
