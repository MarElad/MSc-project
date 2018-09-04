# Deep Learning for Super Resolution Microscopy

In this project we developed a deep learning method to analyse SMLM simulated data in order to generate pixel level predictions for the classificaiton of proteins and their counting.


## Prerequisites

Python packages: Pytorch, Pyvision, Keras, TensorFlow


## Scripts
ResNet18_temporally_uncorrelated_data.ipynb: Residual Network model with 18 layers. Temporally uncorrelated frames' data is included in the code.
ResNet18_temporally_correlated_data.ipynb: Residual Network model with 18 layers. Temporally correlated frames' data is included in the code.
ResNet18_conv3D.ipynb: Residual Network model with 18 3D convolutional layers . Temporally correlated frames' data is included in the code.
Spatial_Renet18_LSTM.py: Residual Network model with 18 2D convolutional layers followed by a LSTM network. Temporally correlated frames' data is included in the code.


## Acknowledgments

* Thanks to Y. Choi for providing the building blocks for the code: https://github.com/yunjey/pytorch-tutorial.

