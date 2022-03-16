# HW3 CNN 
## Task description:
Use CNN model to do the classification task with food11 dataset.

I use the efficientnet_b7 model to reach the boss baseline.<br>
I do the data augmentation. <br>
---------------------------------------

ex:<br>
transforms.RandomVerticalFlip(p=0.3),<br>
transforms.RandomHorizontalFlip(p=0.3),<br>
transforms.RandomGrayscale(p=0.3),<br>
transforms.RandomRotation(degrees=(0, 30)),<br>
transforms.RandomCrop(size=(80,80)),<br>
transforms.RandAugment(),<br>
transforms.RandomPerspective(distortion_scale=0.6, p=0.3),<br>
transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
<br>

---------------------------------------
I use the LabelSmoothingLoss, and do the Test TimeAugmentation(0.3 x train + 0.7 x test)<br>
Finally, I ensemble different model's predictions for getting the best score.

## How to Run
open Anaconda Prompt(Anaconda3)<br>
Run the command below:
```
activate <environment_name>
cd /d<direction>
python Hw3.py
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


