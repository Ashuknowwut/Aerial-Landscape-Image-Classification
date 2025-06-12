import os
import cv2 # install with the command:  pip install opencv-python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score,precision_score,f1_score,recall_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

categories = ['Agriculture','Airport','Beach','City','Desert','Forest','Grassland','Highway','Lake','Mountain','Parking','Port','Railway','Residential', 'River']

def load_data(data_path):
    # load training data
    images = []
    labels = []
    for category in categories:
        path = os.path.join(data_path, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(categories.index(category))
    # convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    # data normalization
    images = images / 255.0
    # reshape data
    n_samples = len(images)
    data = images.reshape((n_samples, -1))
    return data, labels

X_train, y_train = load_data('./Aerial_Landscapes_split/train')
X_test, y_test = load_data('./Aerial_Landscapes_split/test')

# reduce feature dimension using PCA
pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# build KNN model
knn_pca = KNeighborsClassifier(n_neighbors=10)
# fit model
knn_pca.fit(X_train_pca, y_train)
# test model
y_pred = knn_pca.predict(X_test_pca)

# understand the differences among Accuracy, Precision, Recall and F1-score:
# https://zhuanlan.zhihu.com/p/147663370
# https://zhuanlan.zhihu.com/p/405658103
accuracy = accuracy_score(y_test, y_pred)
print('test accuracy: ', accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
print('test precision: ', precision)
recall = recall_score(y_test, y_pred, average='weighted')
print('test recall: ', recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print('test F1 score score: ', f1)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                               display_labels=categories)
disp.ax_.set_xticklabels(categories, rotation=270)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('Confusion_Matrix_knn.png')
plt.show()
plt.close()

# reports
print(classification_report(y_test, y_pred, target_names=categories))

""" output """
# test accuracy:  0.38333333333333336
# test precision:  0.381934600316975
# test recall:  0.38333333333333336
# test F1 score score:  0.3499602952056537
#               precision    recall  f1-score   support
#
#  Agriculture       0.30      0.36      0.33       160
#      Airport       0.31      0.30      0.31       160
#        Beach       0.72      0.53      0.61       160
#         City       0.32      0.21      0.26       160
#       Desert       0.68      0.96      0.80       160
#       Forest       0.39      0.76      0.51       160
#    Grassland       0.33      0.87      0.48       160
#      Highway       0.17      0.11      0.13       160
#         Lake       0.66      0.39      0.49       160
#     Mountain       0.19      0.12      0.15       160
#      Parking       0.31      0.47      0.37       160
#         Port       0.50      0.16      0.25       160
#      Railway       0.31      0.34      0.33       160
#  Residential       0.25      0.08      0.12       160
#        River       0.28      0.07      0.12       160
#
#     accuracy                           0.38      2400
#    macro avg       0.38      0.38      0.35      2400
# weighted avg       0.38      0.38      0.35      2400
