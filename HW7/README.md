# HW7 BERT - Question Answer 
## Task description:
Use Self-Supervised Learning pretrained model to complete the Extractive QA task.

I use the Chinese-Macbert-large model to reach the Boss baseline.<br>
Macbert is similar with bert, but it improves the pretrained process.<br>

--------------------------------------------------------------
#### I use some methods to reinforce my performance : <br>
* I use more limitations to decide the final start / end token of the answer.<br>
* I overcome the training problem on some paragraphs which are longer than max_paragraph_len.<br>
* I adjust the train_validation ratio to get more training data.<br>
* I modified the "hidden_dropout_prob" parameter in config.json for avoiding over-fitting problem.<br>
* I do the post-process to restore the original tokens.
* Finally, I ensemble different model's predictions for getting the best score.
--------------------------------------------------------------

## How to Run
Open the Google Colab<br>
upload the 「ML2022Spring_HW7_ipynb」的副本.ipynb<br>
--------------------------------------------------------------
If you have already obtained Three predictions.<br>
Run the command below<br>
(Remember to modify the code to right directory of prediction.)
```
python post_preprocess.py
python transform_result.py
```
## Replenishment
* You can't directly run the code, download the data is required.
* Watch out the file directory.
* You need to ensemble more than three predictions with 82%~83% accuracy for getting higher score.
* Macbert model ref : https://huggingface.co/hfl/chinese-macbert-large

