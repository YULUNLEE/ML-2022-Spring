# HW10 Attack
## Task description:
Attack objective:Non-targeted attack<br>
Attack constraint:L-infinity norm and parameter epsilon<br>
Attack algorithm:FGSM/I-FGSM/MI-FGSM/DIM+MI-FGSM<br>
I use the DIM+MI-FGSM and diverse model ensemble to reach the Boss baseline.<br>

--------------------------------------------------------------
#### Parameter selection : <br>
* resize_rate = 0.95 (DIM)<br>
* diversity_prob = 0.5 (DIM)<br>
* num_iter = 40 (MI-FGSM)<br>
* decay factor = 1.0 (MI-FGSM)<br>
* epsilon = 8/255/std (MI-FGSM)<br>
* alpha = 0.8/255/std (MI-FGSM)<br>
* model_name = ['resnet56_cifar10',
         'resnet110_cifar10',
         'resnet272bn_cifar10',
         'resnet542bn_cifar10',
         'preresnet56_cifar10',
         'preresnet110_cifar10',
         'preresnet164bn_cifar10',
         'resnext29_32x4d_cifar10',
         'resnext29_16x64d_cifar10',
         'seresnet56_cifar10',
         'seresnet110_cifar10',
         'sepreresnet20_cifar10',
         'sepreresnet56_cifar10',
         'pyramidnet110_a48_cifar10',
         'pyramidnet110_a84_cifar10',
         'diaresnet56_cifar10',
         'diaresnet110_cifar10',
         'diapreresnet56_cifar10',
         'diapreresnet110_cifar10']
--------------------------------------------------------------

## How to Run
Open the Google Colab<br>
upload the 「ML2022Spring_HW10_ipynb」的副本.ipynb<br>
--------------------------------------------------------------
## Replenishment
* You can't directly run the code, download the data is required.
* Watch out the file directory.
* MI-FGSM ref : https://github.com/DengpanFu/RobustAdversarialNetwork/blob/master/pgd_attack.py
* DIM ref : https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/difgsm.py
