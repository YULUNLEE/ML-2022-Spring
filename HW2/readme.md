#HW2 Classification
##Task description:
Extract MFCC features from raw waveform.

I use the RNN model named LSTM to reach the framewise phoneme classification.<br>
Concatenating 91 frames and Using LSTM with 5 hidden Layers, 512 hidden dim.<br>
Furthermore, I add the attention mechanism between the LSTM and FC.<br>
Finally, I ensemble three different model's predictions for getting the best score.

## How to Run
open Anaconda Prompt(Anaconda3)<br>
Run the command below:
```
activate <environment_name>
cd /d<direction>
python Hw2.py
```
If you have already obtained Three predictions.
Run the command below<br>
(Remember to modify the code to right directory of prediction.):
```
python ensemble.py
```
## Replenishment
* You can't directly run the code, download the data is required.
* You need to ensemble more than three predictions with 82% accuracy for getting higher score.


