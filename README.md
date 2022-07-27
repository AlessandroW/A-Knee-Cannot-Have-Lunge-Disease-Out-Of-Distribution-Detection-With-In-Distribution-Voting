# A knee cannot have lung disease: out-of-distribution detection with in-distribution voting using the medical example of chest X-ray classification

This repository includes the official implementation of "A knee cannot have lung disease: out-of-distribution detection with in-distribution voting using the medical example of chest X-ray classification" by Alessandro Wollek, Theresa Willem, Michael
Ingrisch, Bastian Sabel and Tobias Lasser. This method is an effective OOD detector for multi-label classification problems. 

## Pre-trained Models
The pre-trained models can be found in `models/`.

## Test Results
All test results are calculated in the according Jupyter notebooks in `notebooks/`.

## Training
To train the models from scratch you need the following data sets and modify the respective data loaders according to the file structure.:
- [Chest X-Ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC/)
- [IRMA](https://publications.rwth-aachen.de/record/667225)
- [MURA](https://stanfordmlgroup.github.io/competitions/mura/)
- [Bone Age](https://www.kaggle.com/kmader/rsna-bone-age)
- [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge)
