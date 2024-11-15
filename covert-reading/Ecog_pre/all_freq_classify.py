'''
Author: pingping yang
Date: 2024-08-14 05:04:07
Description: 使用SVM将Ecog数据进行二分类
'''
import numpy as np
from ecog_band.datasetAllband import SVMDataset
from ecog_band.models import SVMBinClassification
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ecog_band.utils import *
from ecog_band.solver import Nfold_solver

HS = 86
freq = 500
elec = 7
path_elec = f'/public/DATA/overt_reading/dataset/HS{HS}/{freq}/{elec}'
num_samples = len(os.listdir(path_elec))

data_loader = SVMDataset(HS, path_elec, freq, elec, num_samples)
# for batch in data_loader:
#     batch[0] = batch[0].reshape(-1)
print(len(data_loader)) # 720
for batch in data_loader:
    print(batch[0].shape) # 187875
    break

data, labels = data_loader.get_data_labels()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=1/6, random_state=42)

tuned_parameters = [{'kernel': ['rbf', 'linear'], 'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}]
tuned_parameters_simple = [{'kernel': ['linear'], 'C': [1]}]
cv = 3
scoring = 'accuracy'
svm = SVMBinClassification(tuned_parameters_simple, cv, scoring)

best_params = svm.train(x_train=X_train, y_train=y_train)
print("Best parameters set found on development set:")
print(best_params)

y_pred = svm.evaluate(X_test=X_test, y_test=y_test)
print("Detailed classification report:")
print(classification_report(y_test, y_pred))
print("Accuracy on test set:")
print(accuracy_score(y_test, y_pred))

# plt confusion matrix
plt_confusion_matric(y_test, y_pred)