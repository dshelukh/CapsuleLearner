# CapsuleLearner
Education project to try capsules in neural nets with tensorflow

Run capsule network based on J. Hinton's paper (https://arxiv.org/abs/1710.09829)  
```
python Trainer.py
```

//Under construction: run semi-supervised learning network with capsules on svhn dataset  
```
python SemiSupervised.py [-h] [--mode {ae,semi,both}] [-b B] [-l L] [--save SAVE]  
```
optional arguments:  
*--mode {ae,semi,both}* semi - run semi-supervised network only, ae - run autoencoder part only, both - run autoencoder and convert it to semi-supervised network  
*-b* batch size to use  
*-l* number of labels to use in training  
*--save* specify folder to save to  