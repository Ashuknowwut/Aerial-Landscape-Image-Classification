# COMP9517_Group_project Group One

In this repo, we are exploring [Aerial Landscape Images](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset) dataset. Two methods of feature descriptor plus classifier, three machine learning and five deep learning models are experimented for the landscape classification task.


### Model Description

All models are in file folder `models`, each of them are experimented separately in corresponding file and using a same preprocessed data set (shuffled and splitted) in `Aerial_Landscapes_split`.

Two feature descriptor used with ML classifiers:

- LBP: Local Binary Patterns feature descriptor + classifier
- SIFT: Scale-Invariant Feature Transform + classifier

Three machine learning models:

- KNN: k-Nearest Neighbors	
- RandomForest
- SVM: Support Vector Machine

And five Deep learning models:

- ResNet: Residual Network
- EfficientNet: Efficient Network	
- VGG11: Visual Geometry Group Network
- MLP: Multilayer Perceptron
- MobileNet: Mobile Network


### Reference

- SVM
  source: https://www.kaggle.com/code/ashutoshvarma/image-classification-using-svm-92-accuracy
- VGG
  source: https://github.com/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb
- Mobile Net
  source: https://zhuanlan.zhihu.com/p/147663370
          https://zhuanlan.zhihu.com/p/405658103
- KNN
  source: https://zhuanlan.zhihu.com/p/147663370
          https://zhuanlan.zhihu.com/p/405658103
