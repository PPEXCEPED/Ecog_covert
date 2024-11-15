
# -*- coding: utf-8 -*-
# @Time : 2023/11/18 19:02
# @Author : Zhenjie Wang
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .HS_block_data_pre_process import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as scio
import os
import tqdm
import seaborn as sns
import matplotlib
from tqdm import trange
import pandas as pd
import itertools
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
from scipy.signal import find_peaks, peak_prominences
from sklearn.cluster import KMeans

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.size'] = 10
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设你的数据字典如下
# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 3)
        self.fc3 = nn.Linear(3, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 5折交叉验证

plt.rcParams['axes.unicode_minus']=False
marker_type = {}
marker_type["68"] = "D"
marker_type["69"] = "*"
marker_type["75"] = "s"
marker_type["79"] = "p"
marker_type["82"] = "o"
marker_type["83"] = "v"
marker_type["84"] = "^"
marker_type["85"] = "X"
marker_type["86"] = "P"
class HS_block_process(HSblock_pre_process):
    def __init__(self,HS,workspace_path):
        super().__init__(HS,workspace_path)
        # print(self.block_path+"/HS"+str(self.HS)+"block.npy")
        self.HSblock = np.load(self.block_path+"/HS"+str(self.HS)+"block_500.npy",allow_pickle=True ).item()

    def get_HSblock(self):
        return self.HSblock

    # def SVM(self):
    
def get_task_word(task_list=["cue",'reading','listen']):
    """
    获取任务词列表
    """
    a = []
    keys = ['功课','树叶', '力果','对十' , '宫客','数页', '作业','绿草', 'gōng kè','shù yè']
    

    for task in task_list:
        for word in keys:
            if task=='listen' and word in [ '宫客', 'gōng kè','数页','shù yè']:
                pass
            else:
                a.append(f"{task}_{word}")

    return a
def get_HS_elec(subjects,sig_elecs_HS_lists,task_list=["cue","reading","listen"]):
    """
    获取HS的电极
    """
    HS_elec = []
    for HS in subjects:
        
        elec_list = []
        for task in task_list:
            for i in sig_elecs_HS_lists[f'HS{HS}'][task]:
                if i not in elec_list:
                    elec_list.append(i)
                    
                    HS_elec.append(f"{HS}_{i}")
    return HS_elec
                

def train_svm_classifier(X, y):
    """
    训练一个 svm 分类器。
    
    参数:
    X -- 特征数据
    y -- 标签数据

    返回:
    model -- 训练好的模型
    accuracy -- 在测试集上的准确率
    """
    # 将数据集分为训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 SVM 模型
    
    model = SVC(kernel='linear', C=1e5)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def two_class_classifier(X, y,only_acc=False,model="xgboost"):
    """
    训练一个 XGBoost 分类器。
    
    参数:
    X -- 特征数据
    y -- 标签数据

    返回:
    model -- 训练好的模型
    accuracy -- 在测试集上的准确率
    """
    device = torch.device("cpu")

    # 将数据集和测试集按照五折交叉验证来划分
    # 使用分层抽样进行五折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if only_acc:
        accuracy = []
    else:
        accuracy = []
        report = []
        cm = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    

        # 训练 XGBoost 模型
        if model == "xgboost":
            model = xgb.XGBClassifier(eval_metric='mlogloss')
            model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = model.predict(X_test)

            # 计算准确率
            if only_acc:
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
            else:
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
                # 返回混淆矩阵
                cm.append(confusion_matrix(y_test, y_pred))
                report.append(classification_report(y_test, y_pred))
        elif model == "mlp":
            # 使用torch的resnet50作为模型
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            X_val_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
            
            # 数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            
            # 初始化模型、损失函数和优化器
            model = SimpleNN().to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练模型
            for epoch in range(10):  # 简单起见，我们只训练10个epoch
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 验证模型
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu()
                    batch_y = batch_y.cpu()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                        
            accuracy.append(correct / total)
        elif model == "logistic":
            model = LogisticRegression(solver='lbfgs', max_iter=10000)
            model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = model.predict(X_test)

            # 计算准确率
            if only_acc:
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
            else:
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
                # 返回混淆矩阵
                cm.append(confusion_matrix(y_test, y_pred))
                report.append(classification_report(y_test, y_pred))
    if only_acc:
        return  accuracy
    else:
        return model,accuracy,report,cm


def get_timelocked_activity(times, hg, hz=100, back=20, forward=100):
    times = np.array(times)
    times = times[times * hz - back > 0]
    times = times[times * hz + forward < hg.shape[1]]
    Y_mat = np.zeros((hg.shape[0], int(back + forward), len(times)), dtype=float)

    for i, seconds in enumerate(times):
        index = int(np.round(seconds * hz))
        Y_mat[:, :, i] = hg[:, int(index-back):int(index+forward)]

    return Y_mat

def reshape_grid(x, channel_order=None):
    """Takes an array of 256 values and returns a 2d array that can be matshow-ed
    x -- array or list of values.
    """
    x = np.array(x)[:256]
    if channel_order is None:
        channel_order = np.array([256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,254,238,222,206,190,174,158,142,126,110,94,78,62,46,30,14,253,237,221,205,189,173,157,141,125,109,93,77,61,45,29,13,252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12,251,235,219,203,187,171,155,139,123,107,91,75,59,43,27,11,250,234,218,202,186,170,154,138,122,106,90,74,58,42,26,10,249,233,217,201,185,169,153,137,121,105,89,73,57,41,25,9,248,232,216,200,184,168,152,136,120,104,88,72,56,40,24,8,247,231,215,199,183,167,151,135,119,103,87,71,55,39,23,7,246,230,214,198,182,166,150,134,118,102,86,70,54,38,22,6,245,229,213,197,181,165,149,133,117,101,85,69,53,37,21,5,244,228,212,196,180,164,148,132,116,100,84,68,52,36,20,4,243,227,211,195,179,163,147,131,115,99,83,67,51,35,19,3,242,226,210,194,178,162,146,130,114,98,82,66,50,34,18,2,241,225,209,193,177,161,145,129,113,97,81,65,49,33,17,1])-1
    x_reordered = x[channel_order]
    return np.reshape(x_reordered, (16,16))

def get_mean_and_ste(to_average):
    """Takes chans x timepoints x trials and averages over trials returning chans x timepoints
    This function returns the average and ste over the third dimension (axis=2) and returns
    an array of the first two dimensions
    """
    average = np.nanmean(to_average, axis=2)
    ste = np.nanstd(to_average, axis=2)/np.sqrt(np.shape(to_average)[2])

    min_value = np.nanmin([average-ste])
    max_value = np.nanmax([average+ste])
    return average, ste, min_value, max_value

def create_grid_fig(highlight_channels=np.array([1000]), bcs=None, hlines=None, vlines=None, channel_order=None, background_colors=None, n_chans=256):
    fig = plt.figure()
    if channel_order is None:
        channel_order = [256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,254,238,222,206,190,174,158,142,126,110,94,78,62,46,30,14,253,237,221,205,189,173,157,141,125,109,93,77,61,45,29,13,252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12,251,235,219,203,187,171,155,139,123,107,91,75,59,43,27,11,250,234,218,202,186,170,154,138,122,106,90,74,58,42,26,10,249,233,217,201,185,169,153,137,121,105,89,73,57,41,25,9,248,232,216,200,184,168,152,136,120,104,88,72,56,40,24,8,247,231,215,199,183,167,151,135,119,103,87,71,55,39,23,7,246,230,214,198,182,166,150,134,118,102,86,70,54,38,22,6,245,229,213,197,181,165,149,133,117,101,85,69,53,37,21,5,244,228,212,196,180,164,148,132,116,100,84,68,52,36,20,4,243,227,211,195,179,163,147,131,115,99,83,67,51,35,19,3,242,226,210,194,178,162,146,130,114,98,82,66,50,34,18,2,241,225,209,193,177,161,145,129,113,97,81,65,49,33,17,1]
        channel_labels = channel_order
    else:
        n_chans = len(channel_order)
        if 255 in channel_order:
            channel_labels = channel_order
            channel_order = [c - 128 + 1 for c in channel_order]

    if n_chans == 256:
        n_rows, n_cols = (16, 16)
        fig.set_size_inches(20, 20)
    elif n_chans == 128:
        n_rows, n_cols = (8, 16)
        fig.set_size_inches(20, 10)

    if bcs is None:
        bcs = []
    axs = []
    for i in range(n_chans):
        if i not in bcs:
            ax = fig.add_subplot(n_rows, n_cols, channel_order[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if hlines is not None:
                for hline in hlines:
                    ax.axhline(y=hline, color='gray', alpha=0.5)
            if vlines is not None:
                for vline in vlines:
                    ax.axvline(x=vline, color='gray', alpha=0.5)
            if i in highlight_channels:
                for side in ['bottom', 'top', 'left', 'right']:
                    ax.spines[side].set_color('black')
            if background_colors is not None:
                ax.set_facecolor(background_colors[i])
            if n_chans == 256:
                ax.text(0.2,0.8,str(i),transform=ax.transAxes)
            elif n_chans == 128:
                ax.text(-0.1, 0.85,str(i+128),transform=ax.transAxes)
                ax.axis("off")
            axs.append(ax)
        else:
            axs.append(None)
    return fig, axs

def add_to_grid_fig(axs, xvals, yvals, color, ylim=(0,1), xlim=None, plot_kws={}):
    n_chans = len(yvals)
    for i in range(n_chans):
        ax = axs[i]
        if ax is not None:
            ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.plot(xvals, yvals[i], color=color, **plot_kws)
    return axs

def add_erps(axs, times, hg, color, plot_kws={}, hz=100, back=25, forward=275, bcs=None, xticks=[25, 245]):
    n_chans = 256
    if bcs is None:
        bcs = []
    hg_to_average = np.zeros((n_chans, back+forward, np.shape(times)[1]-1), dtype=float)
    count = 0
    for i, seconds in enumerate(times[0]):
        if i>0:
            index = np.round(seconds * hz)
            hg_to_average[:,:,count] = hg[:,index-back:index+forward]
            count += 1
    average_hg = np.nanmean(hg_to_average,axis=2)
    ste_hg = np.nanstd(hg_to_average, axis=2)/np.sqrt(np.shape(hg_to_average)[2])
    min_value = np.min(average_hg)
    max_value = np.max(average_hg)
    print(min_value)
    print(max_value)
    for i in range(n_chans):
        if i not in bcs:
            ax = axs[i]
            ax.plot(average_hg[i], color=color, **plot_kws)
            ax.set_ylim(min_value, max_value)
            ax.set_yticks([min_value, max_value])
    return axs

def plot_erps(HS_BLOCK,HS_elecs_list,tasks_list,ax):
    a = []
    for task in tasks_list:
        for HS_elec in HS_elecs_list:
            HS,elec = HS_elec.split('_')
            elec = int(elec)
            a.append(HS_BLOCK[f"HS{HS}"][task][elec])
    y = np.concatenate(a,-1)
    x_val = range(y.shape[0])
    if ax == None:
        plt.figure()
        plt.plot(x_val,y)
        plt.fill_between(x_val,y-np.std(y)/np.sqrt(y.shape[1]),y+np.std(y)/np.sqrt(y.shape[1]),alpha=0.2)
        plt.plot(x_val,1/2*np.ones_like(x_val))
    else:
        # 从plt改成ax形式
        ax.plot(x_val,y)
        ax.fill_between(x_val,y-np.std(y)/np.sqrt(y.shape[1]),y+np
                        .std(y)/np.sqrt(y.shape[1]),alpha=0.2)
        ax.plot(x_val,1/2*np.ones_like(x_val))

    return 