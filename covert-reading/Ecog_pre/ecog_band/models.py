import torchvision
from torchvision import models
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from ecog_band.utils import plt_learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA


class ECOGRes50(nn.Module):
    def __init__(self):
        super(ECOGRes50, self).__init__()
        # self.conv7to1 = nn.Conv2d(7, 1, kernel_size=1, stride=1,padding=0, bias=False)
        self.res50= models.resnet50()
        self.res50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.res50.fc.in_features
        self.res50.fc = nn.Sequential(nn.Linear(num_ftrs,1, bias=False),
                                    )
        
    def forward(self,x):
        # x = self.conv7to1(x)
        x=self.res50(x)
        return x

class EcogBandRes(nn.Module):
    def __init__(self):
        super(EcogBandRes,self).__init__()
        self.res=ECOGRes50()

    def forward(self,x):
        # x=self.linearmap(x)
        x=self.res(x)

        return x


class ECOGRes50_feature(nn.Module):
    def __init__(self):
        super(ECOGRes50_feature, self).__init__()
        self.res50 = models.resnet50()
        self.res50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res50.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP Layer
        num_ftrs = self.res50.fc.in_features
        self.res50.fc = nn.Linear(num_ftrs, 1, bias=False)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.res50(x)
        return x
    
    def get_cam_weights(self):
        return self.res50.fc.weight.data.squeeze()

    def get_features(self, x):
        # Get the output of the last convolutional layer
        x = self.res50.conv1(x)
        x = self.res50.bn1(x)
        x = self.res50.relu(x)
        x = self.res50.maxpool(x)
        x = self.res50.layer1(x)
        x = self.res50.layer2(x)
        x = self.res50.layer3(x)
        x = self.res50.layer4(x)
        x = self.dropout(x)
        features= self.res50.fc(x)
        return features
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(8, 3)  # Assuming there are 3 classes

    def forward(self, x):
        x = self.linear(x)
        return x

class SVMBinClassifier(nn.Module):
    def __init__(self):
        # tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1], 'gamma': [0.01]}]
        # self.model = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=1)
        self.model = SVC(C=0.1, kernel='sigmoid')
        self.pca = PCA(n_components=0.95)
    
    def train(self, x_train, y_train):
        X_train_pca = self.pca.fit_transform(x_train)
        self.model.fit(X_train_pca, y_train)
        # plt_learning_curve(self.model, X_train_pca, y_train, cv=5)
        # best_params = self.model.best_params_
        # return best_params
    
    def predict(self, x_test):
        X_test_pca = self.pca.transform(x_test)
        return self.model.predict(X_test_pca)
    
    def evaluate(self, X_test, y_test):
        # X_test_pca = self.pca.transform(X_test)
        y_pred = self.predict(X_test)
        # y_pred = cross_val_predict(self.model, X=X_test_pca, y=y_test, cv=5)
        return y_pred

# class SVMBinClassifier:
#     def __init__(self, tuned_parameters=None):
#         if tuned_parameters is None:
#             tuned_parameters = [{'kernel': ['linear', 'rbf', 'sigmoid'], 
#                                  'C': [0.1, 1, 10], 
#                                  'gamma': [0.01, 0.1, 1]}]
        
#         self.tuned_parameters = tuned_parameters
#         self.model = None  # 模型将由 GridSearchCV 创建
#         self.pca = PCA(n_components=0.95)  # PCA设置为保留95%方差

#     def train(self, x_train, y_train):
#         # PCA降维
#         X_train_pca = self.pca.fit_transform(x_train)
        
#         # 使用GridSearchCV进行超参数搜索
#         self.model = GridSearchCV(SVC(), self.tuned_parameters, scoring='accuracy', cv=5, n_jobs=-1)
#         self.model.fit(X_train_pca, y_train)
        
#         # 输出最优参数
#         best_params = self.model.best_params_
        
#         return best_params  # 返回最佳超参数

#     def predict(self, x_test):
#         # 使用PCA对测试数据降维
#         X_test_pca = self.pca.transform(x_test)
        
#         # 使用最优模型进行预测
#         return self.model.predict(X_test_pca)

#     def evaluate(self, X_test, y_test):
#         # 预测测试集
#         y_pred = self.predict(X_test)
        
#         return y_pred


class DecisionTreeBinClassifier(nn.Module):
    def __init__(self):
        super(DecisionTreeBinClassifier, self).__init__()
        self.pca = PCA(n_components=0.95)
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, max_features='sqrt', max_leaf_nodes=20)

        # # Hyperparameter grid for tuning
        # self.param_grid = {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }
        # self.grid_search = GridSearchCV(self.model, self.param_grid, cv=5, n_jobs=-1)

    def train(self, x_train, y_train):
        X_train_pca = self.pca.fit_transform(x_train)
        self.model.fit(X_train_pca, y_train)
        # self.grid_search.fit(X_train_pca, y_train)
        # self.model = self.grid_search.best_estimator_
        # plt_learning_curve(self.model, x_train, y_train)

    def predict(self, x_test):
        X_test_pca = self.pca.transform(x_test)
        return self.model.predict(X_test_pca)

    def evaluate(self, X_test, y_test):
        # X_test_pca = self.pca.transform(X_test)
        y_pred = self.predict(X_test)
        # y_pred = cross_val_predict(self.model, X=X_test, y=y_test, cv=5)
        return y_pred
    

class RandomForestBinClassifier(nn.Module):
    def __init__(self):
        # param_grid = {
        #     'n_estimators': [50],
        #     'max_depth': [10]
        # }
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10)
        # self.model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        # best_params = self.model.best_params_
        # return best_params
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = cross_val_predict(self.model, X=X_test, y=y_test, cv=5)
        # y_pred = self.predict(X_test)
        return y_pred
    
class KNeighborsBinClassifier(nn.Module):
    def __init__(self):
        params = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        self.model = GridSearchCV(KNeighborsClassifier(), params, cv=5)
        # self.model = KNeighborsClassifier()
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        best_params = self.model.best_params_
        return best_params
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return y_pred

class GaussianNBBinClassifier(nn.Module):
    def __init__(self):
        # param_grid = {
        #     'priors': [None, [0.2, 0.3, 0.5]],  
        #     'var_smoothing': [1e-10, 1e-9, 1e-8] 
        # }
        # self.model = GridSearchCV(GaussianNB(), param_grid, cv=5)
        self.model = GaussianNB()
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        # best_params = self.model.best_params_
        # return best_params
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return y_pred
    
def extract_features(stft_block,freq,best_fold_model_path,elec):
    bands = {
        'else1': (0,1),
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 150),
        'else2':(150,freq+1)
    }

    features = []
    for band_each, (low, high) in bands.items():
        model_files = [f for f in os.listdir(os.path.join(best_fold_model_path, str(elec))) if f.endswith(f"{band_each}.pth")]
        assert len(model_files) == 1, f"Model for band {band_each} not found or multiple models found."
        model_file = model_files[0]

        net = ECOGRes50_feature()
        net.load_state_dict(torch.load(os.path.join(best_fold_model_path, str(elec), model_file)))

        f=torch.arange(stft_block.shape[0])
        indices = np.where((f >=low) & (f < high))
        stft_block=stft_block[indices, :]
        print(stft_block.shape)

        with torch.no_grad():
            feature = net.get_features(stft_block)  # 加上batch维度
            features.append(feature)
            print(feature.shape)

    return features