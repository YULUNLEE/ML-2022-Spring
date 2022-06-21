# HW13 Network Compression
## Task description:
![screenshot-miniimagetnet](report_Q3.png)

* Network Compression: Make your model smaller without losing the performance.<br>
* Knowledge Distillation - Train a small model(student model) with teacher model, small model can learn better.<br>
* Use Depthwise and Pointwise Convolution to reduce the model parameters.(parameters<10k)<br>
* Train an image classifier to classify the kind of food. <br>
* Network pruning <br>

---------------------------------------
## Some methods to be used: <br>
* Residual learning
* move some validation set to training set --> train : 12796, valid : 500
* Using data augmentation like HW3, instead of putting all Data augmentation methods in begin, put one by one in gradually.
---------------------------------------
## How to Reproduce the best result
Open the Google Colab<br>
Upload the「ML2022Spring_HW13_ipynb」的副本 <br>

## Replenishment
* You can't directly run the code, download the data is required.
* Watch out the file directory, and have to load the checkpoint successfully.
