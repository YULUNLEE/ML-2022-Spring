# HW4 self-Attention 
## Task description:
Use self-Attention mechanism to do the classification task with VoxCeleb2 dataset. <br>
input a voice --> recognize who is speaking

I use the Conformer model to reach the task.<br>
Conformer is a variant of Transformer, it uses the convolutional layer to boost the performance.<br>

I use the self-Attention Pooling before enter the FC.<br>
Consider the Longer length of voice.<br>
Finally, I ensemble different model's predictions for getting the best score.

## How to Run
open Anaconda Prompt(Anaconda3)<br>
Run the command below:
```
activate <environment_name>
cd /d<direction>
python Hw4.py
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
* You need to ensemble more than three predictions with 88% accuracy for getting higher score.


