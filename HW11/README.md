# HW11 Adaptation
## Task description:
Domain Adaptation　：<br>
Given real images (with labels) and drawing images (without
labels), please use domain adaptation technique to make your network
predict the drawing images correctly.

--------------------------------------------------------------
#### I use some methods to reinforce my performance : <br>
* epoch = 2000, train more time.
* I implement the get_Lambda function, the Lambda value will gradually increase at every epoch.<br>
* I implement the adjust_learning_rate function, it's a LearningRate scheduler which is defined in the paper. <br>
--------------------------------------------------------------

## How to Run
open Anaconda Prompt(Anaconda3)<br>
Run the command below :<br>
```
activate <environment_name>
cd /d<direction>
python hw11.py
```
If you have already obtained Three predictions.
Run the command below<br>
(Remember to modify the code to right directory of prediction.)
```
python ensemble.py
```
## Replenishment
* You can't directly run the code, download the data is required.
* Watch out the file directory.
* You need to ensemble more than three predictions with 80%~81% accuracy for getting higher score.
* Domain-Adversarial Training of Neural Networks paper ref : https://arxiv.org/pdf/1505.07818.pdf

