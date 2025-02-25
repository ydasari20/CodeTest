1. Custom CNN for CIFAR-10 Classification
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset.

Requirements
Install torch torchvision scikit-learn seaborn matplotlib

Dataset
The model is trained and tested on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.

Model Overview
A custom CNN architecture with Convolutional layers with batch normalization and ReLU activation, Max pooling for dimensionality reduction, Dropout layers to reduce overfitting, Fully connected layers for classification.
Uses CrossEntropyLoss as the loss function.
Optimized using Adam optimizer.

Training the Model
python train.py is used to train the model.
The training loop runs for 20 epochs with a batch size of 64 .

Evaluating the Model
After training, the model evaluates performance on the test set and prints the accuracy. It also generates a confusion matrix plot.

Saving athe Model
After training, the model is saved as custom_cnn.pth


Fine-Tuning ResNet-18 for Binary Classification
This is fine-tuning a pretrained ResNet-18 model for binary classification using PyTorch. The model is trained on a toy dataset and can be adapted for custom datasets.

Requirements - Torch (Pytorch)

Dataset - The script currently uses torchvision.datasets.FakeData as a placeholder dataset. Replace this with your custom dataset as needed.

Model Overview
Loads a pretrained ResNet-18 model.
Retains pretrained features.
Replaces the fully connected (fc) layer for binary classification.
Uses BCEWithLogitsLoss for training.

Training the Model
python train.py is used to train the model.
The training loop runs for 5 epochs with Adam optimizer and a learning rate of 0.001.

Saving the Model
After training, the model is saved as finetuned_resnet18.pth


2. Transfer learning is widely used in computer vision because training deep neural networks from scratch requires massive amounts of labeled data and computational power. Pretrained models, trained on large datasets like ImageNet, learn rich feature representations that can be reused for new tasks. By leveraging these pretrained features, we can significantly reduce training time and improve performance, especially when dealing with smaller datasets.

Fine-tuning only the last few layers allows the model to adapt to the new task without losing the general features learned earlier, striking a balance between learning new patterns and retaining useful knowledge.
