# HW5 Transformer
## Task description:
Use the Transformer to complete English to Chinese translation

I use the Transformer with <br>

-----------------------------------
* encoder/decorder_embed_dim = 1024.<br>
* encoder/decorder_ffn_embed_dim = 4096.<br>
* encoder/decorder_layers = 8.<br>
* Dropout = 0.3<br>
* encoder/decoder_attention_heads = 16<br>
* epoch = 50
* Training time = more than 2 days
-----------------------------------
Because of GPU RAM constrain, I adjust the max_tokens to 2048 and accum_steps to 8.<br>
I use the monolingual data in order to train a backward model <br>
--> it can increase our training data to get a higher score.<br>


## How to Run
Open  the Google Colab<br>
upload the 「HW05_ipynb」的副本.ipynb:<br>


## Replenishment
* You can't directly run the code, download the data is required.
* Watch out the file directory.
* You need to ensemble more than five checkpoints for getting higher score.


