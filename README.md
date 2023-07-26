# CIFAR-10-Random-and-1-NN-Classifier

This project explores image classification using the popular CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images, divided into 10 classes, each containing 6,000 images. The goal is to classify these images into their respective categories using two classifiers: a Random Classifier and a 1-NN (Nearest Neighbor) Classifier.

The project utilizes Python and NumPy to process and analyze the dataset. It begins by introducing the CIFAR-10 dataset, downloading it, and visualizing random samples from the dataset. It then evaluates the classification accuracy using a custom accuracy function.

The project implements two classifiers. The first classifier, Random Classifier, assigns random class labels to the input data. The second classifier, 1-NN Classifier, finds the best matching training sample from the CIFAR-10 training set and assigns its label to the input data. The accuracy of both classifiers is evaluated using the custom accuracy function.

The project aims to demonstrate the effectiveness of the 1-NN classifier in comparison to the Random Classifier for image classification tasks and to showcase how simple classifiers can perform on complex datasets like CIFAR-10.
