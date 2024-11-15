import os
import itertools
import statistics
from typing import Callable
import numpy as np
# from tqdm import tqdm
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm  # may raise warning about Jupyter
from tqdm.auto import tqdm  # who needs warnings
import torch, torchvision
from torch import nn
from torch.utils import data as Data
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import itertools
import statistics
from typing import Callable
import numpy as np
# import tensorflow as tf
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import copy
from sklearn.model_selection import StratifiedKFold


class Solver(object):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable,
                 lr_scheduler = None,
                 recorder: dict = None,
                 device=None,
                 early_stopping_patience=10):
        device = device if device is not None else \
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.recorder = recorder
        
        self.model = self.to_device(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        #early stop 
        self.best_per = 0 
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')       

    def _step(self,
              batch: tuple) -> dict:
        raise NotImplementedError()

    def to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=self.device)
        elif isinstance(x, nn.Module):
            return x.to(self.device)
        else:
            raise RuntimeError("Data cannot transfer to correct device.")

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            raise RuntimeError(f"Cannot convert type {type(x)} into numpy array.")

    def train(self,
              epochs: int,
              data_loader,
              *,
              val_loader=None,
              is_plot=True) -> dict:             
        torch.cuda.empty_cache()

        val_loss_epochs = []
        train_loss_epochs = []
        val_acc=[]
        all_preds = []  # Collect TPR for each epoch
        all_labels = []  # Collect FPR for each epoch

        pbar_train = tqdm(total=len(data_loader)*2-1, unit='block')
        if val_loader is not None:
            pbar_val = tqdm(total=len(val_loader), desc=f'[Validation] waiting', unit='block')

        train_acc=[]
        for epoch in range(epochs):

            pbar_train.reset()
            pbar_train.set_description(desc=f'[Train] Epoch {epoch + 1}/{epochs}')
            epoch_loss_acc = 0
            epoch_size = 0
            epoch_acc=0
    
            
            for batch in data_loader:
                self.model.train()
                # forward
                step_dict = self._step(batch)
                batch_size = step_dict['batch_size']
                loss = step_dict['loss']
                acc=step_dict['acc']
                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # optimize
                self.optimizer.step()

                # update information
                loss_value = loss.item()
                epoch_loss_acc += loss_value
                epoch_size += batch_size
                epoch_acc+=acc

                pbar_train.update(batch_size)
                pbar_train.set_postfix(loss=loss_value / batch_size)

            epoch_avg_loss = epoch_loss_acc / epoch_size
            epoch_avg_acc=epoch_acc/epoch_size
            pbar_train.set_postfix(epoch_avg_loss=epoch_avg_loss)
            train_loss_epochs.append(epoch_avg_loss)
            train_acc.append(epoch_avg_acc)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # validate if `val_loader` is specified
            if val_loader is not None:
                pbar_val.reset()
                pbar_val.set_description(desc=f'[Validation] Epoch {epoch + 1}/{epochs}')
                val_avg_loss,total_avg_acc,TP,TN,FP,FN, epoch_preds, epoch_labels= self.validate(val_loader, pbar=pbar_val,is_test=False)
                val_loss_epochs.append(val_avg_loss)
                val_acc.append(total_avg_acc)

                all_preds.extend(epoch_preds)
                all_labels.extend(epoch_labels)
                
                if total_avg_acc > self.best_per:
                    torch.save(self.model,'best_model.pth')
                    self.best_per = total_avg_acc

            # Early Stopping

                if val_avg_loss < self.best_val_loss:
                    self.best_val_loss = val_avg_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.end_epoch=epoch+1
                    print(f'Early stopping at epoch {self.end_epoch}\n')
                    print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
                    break

        pbar_train.close()
        if val_loader is not None:
            pbar_val.close()
        train_loss_epochs = torch.tensor(train_loss_epochs).numpy()
        val_loss_epochs = torch.tensor(val_loss_epochs).numpy()

        plt.figure()
        plt.plot(list(range(1, self.end_epoch+1)), train_loss_epochs, label='train')
        if val_loader is not None:
            plt.plot(list(range(1, self.end_epoch+1)), val_loss_epochs, label='validation')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close('all')

        if val_loader is not None:
            train_acc=torch.tensor(train_acc).numpy()
            val_acc=torch.tensor(val_acc).numpy()
            plt.figure()
            plt.plot(list(range(1, self.end_epoch+1)), train_acc, label='train')
            plt.plot(list(range(1, self.end_epoch+1)), val_acc, label='validation')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('ACC')
            plt.show()
            plt.close('all')

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def generate_cam(self, features, weights,block):
        '''
        feature: 特征图，通常是CNN中某个层的输出，形状为(batch_size, num_channels, height, width)
        weights: 用于加权特征图的权重，通常是模型的最后一个全连接层的权重
        block: 原始数据块，形状为 (batch_size, num_channels, n_ferq, n_timePoints) 原始数据是stft变换得到的频谱图
        '''
        _, nc, h, w = features.shape
        cam = torch.zeros((h, w), dtype=torch.float32).to(self.device)

        for i in range(nc):
            cam += weights[i].item() * features[0, i, :, :]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

    # Interpolate CAM to the size of the original image
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(block.size(2), block.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        return cam
        
    def validate(self, data_loader, *, pbar=None,is_test=False) -> float:
        sigmoid = nn.Sigmoid()
        """
        :param pbar: when pbar is specified, do not print average loss
        :return:
        """       
        torch.cuda.empty_cache()
        metrics_acc = {}
        loss_acc = 0
        size_acc = 0
        total_acc=0
        is_need_log = (pbar is None)  

        # Initialize TP, TN, FP, FN
        TP, TN, FP, FN = 0, 0, 0, 0  

        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            if pbar is None:
                pbar = tqdm(total=len(data_loader), desc=f'[Validation]', unit='block')

            
            for batch in data_loader:
                self.model.eval()

                block, gt = batch
                block = self.to_device(block)
                gt = self.to_device(gt)

                # forward
                features = self.model.get_features(block)
                pred = self.model(block)  # [B, 1]
                
                if is_test:
                # 获取特征图和权重
                    weights = self.model.get_cam_weights()

                    # 生成CAM
                    cam = self.generate_cam(features, weights,block)

                    # Plot original image and CAM
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(block[0].permute(1, 2, 0).cpu().numpy(),cmap='coolwarm')
                    plt.title('Original Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(cam.cpu().numpy(), cmap='jet')
                    plt.title('CAM')
                    plt.show()

                    print(pred)
                    print(gt)
                    print(sigmoid(pred))
                    
                    pred_mask_num = 0 if sigmoid(pred).item() < 0.5 else 1
                    print(pred_mask_num)
                    print('Prediction is correct' if pred_mask_num == gt.item() else 'Prediction is incorrect')


                # forward
                pred_mask=sigmoid(pred)

                pred_mask[pred_mask>0.5]=1
                pred_mask[pred_mask<=0.5]=0

                all_preds.extend(pred_mask.cpu().numpy())
                all_labels.extend(batch[1].to(torch.int).cpu().numpy())

                acc=torch.sum(pred_mask==batch[1].to(torch.int).to(self.device))

                TP += ((pred_mask == 1) & (batch[1].to(self.device) == 1)).sum().item()
                TN += ((pred_mask == 0) & (batch[1].to(self.device) == 0)).sum().item()
                FP += ((pred_mask == 1) & (batch[1].to(self.device) == 0)).sum().item()
                FN += ((pred_mask == 0) & (batch[1].to(self.device) == 1)).sum().item()


                step_dict = self._step(batch)
                batch_size = step_dict['batch_size']
                loss = step_dict['loss']
                acc=step_dict['acc']
                # print(acc)
                loss_value = loss.item()

                # aggregate metrics
                metrics_acc = self._aggregate_metrics(metrics_acc, step_dict)

                # update information
                loss_acc += loss_value
                size_acc += batch_size
                total_acc+=acc
                pbar.update(batch_size)
                pbar.set_postfix(loss=loss_value)


        val_avg_loss = loss_acc / size_acc
        total_avg_acc=total_acc/size_acc
        if is_test:
            print(total_avg_acc)
        pbar.set_postfix(val_avg_loss=val_avg_loss)
        if is_need_log:
            pbar.close()  # destroy newly created pbar
            print('=' * 30 + ' Measurements ' + '=' * 30)
            for k, v in metrics_acc.items():
                print(f"[{k}] {v / size_acc}")
        else:
            return val_avg_loss,total_avg_acc,TP,TN,FP,FN,all_preds, all_labels
        

    def _aggregate_metrics(self, metrics_acc: dict, step_dict: dict):
        batch_size = step_dict['batch_size']
        for k, v in step_dict.items():
            if k[:7] == 'metric_':
                value = v * batch_size
                metric_name = k[7:]
                if metric_name not in metrics_acc:
                    metrics_acc[metric_name] = value
                else:
                    metrics_acc[metric_name] += value
        return metrics_acc


    def get_recorder(self) -> dict:
        return self.recorder




class LabSolver(Solver):
    def _step(self, batch) -> dict:
        block, gt = batch

        block = self.to_device(block)  # [B, C=1, H, W]
        gt = self.to_device(gt)  # [B, C=1, H, W]
        
        B, C, H, W = block.shape
        # print(block.shape)

        pred = self.model(block)  # [B, C=1, H, W]
        gt = gt.view(B,1)
        loss = self.criterion(pred, gt.to(torch.float32))

        threshold=0.5
        pred_mask=torch.sigmoid(pred)
        pred_mask[pred_mask>threshold]=1
        pred_mask[pred_mask<=threshold]=0
        acc=torch.sum(pred_mask==gt.to(torch.int))
        # mask=torch.argmax(pred_mask,dim=1)
        # one_hot_pred = F.one_hot(mask, num_classes=pred_mask.shape[1])

        # assert one_hot_pred.shape==gt.shape
        # acc=torch.sum(one_hot_pred==gt)/2
    

        step_dict = {
            'loss': loss,
            'batch_size': B,
            'acc':acc
        }

        return step_dict


class Nfold_solver(Solver):
    def __init__(self, model, optimizer, criterion, lr_scheduler=None, recorder=None, device=None, early_stopping_patience=10):
        super().__init__(model, optimizer, criterion, lr_scheduler, recorder, device, early_stopping_patience)

    def train(self,
              epochs: int,
              data_loader,#all the dataset
              band,
              best_fold_model_path,
              *,
              val_loader=None,
              is_plot=True,
              permutation_test=False) -> dict:             
        torch.cuda.empty_cache()
        best_fold_acc=0
        fold_results=[]
        # kf=KFold(n_splits=5) 
        skf = StratifiedKFold(n_splits=5)
        self.initial_model_state = copy.deepcopy(self.model.state_dict())  # Save the initial state
        
        labels = [label for _, label in data_loader]
        fold_all_preds = []
        fold_all_labels = []


        for fold, (train_idx, val_idx) in enumerate(skf.split(data_loader, labels)):
            print(f'Fold {fold + 1}')
            self.model.load_state_dict(copy.deepcopy(self.initial_model_state))  # Reset the model to its initial state
            best_val_loss=float('inf')
            early_stopping_counter=0
            train_acc=[]
            val_loss_epochs = []
            train_loss_epochs = []
            val_acc=[]
            all_preds = []  # Collect TPR for each epoch
            all_labels = []  # Collect FPR for each epoch

            if permutation_test == True:
                labels_tmp = [label for _, label in data_loader]
                permuted_labels = np.random.permutation(labels_tmp)
                data_loader_tmp = [(data, permuted_label) for (data, _), permuted_label in zip(data_loader, permuted_labels)]
                train_subset = Subset(data_loader_tmp, train_idx)
            else:
                train_subset = Subset(data_loader, train_idx)

            val_subset = Subset(data_loader, val_idx)

            train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            

            pbar_train = tqdm(total=len(train_loader)*2, unit='block')
            if val_loader is not None:
                pbar_val = tqdm(total=len(val_loader), desc=f'[Validation] waiting', unit='block')

            for epoch in range(epochs):

                pbar_train.reset()
                pbar_train.set_description(desc=f'[Train] Epoch {epoch + 1}/{epochs}')
                epoch_loss = 0
                epoch_size = 0
                epoch_acc=0
                self.end_epoch=epoch+1
                
                for batch in train_loader:
                    self.model.train()
                    # forward
                    step_dict = self._step(batch)
                    batch_size = step_dict['batch_size']
                    loss = step_dict['loss']
                    acc=step_dict['acc']

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()

                    # optimize
                    self.optimizer.step()

                    # update information
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    epoch_size += batch_size
                    epoch_acc+=acc

                    pbar_train.update(batch_size)
                    pbar_train.set_postfix(loss=loss_value / batch_size)

                epoch_avg_loss = epoch_loss / epoch_size
                epoch_avg_acc=epoch_acc/epoch_size
                pbar_train.set_postfix(epoch_avg_loss=epoch_avg_loss)
                train_loss_epochs.append(epoch_avg_loss)
                train_acc.append(epoch_avg_acc)

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # validate if `val_loader` is specified
                if val_loader is not None:
                    pbar_val.reset()
                    pbar_val.set_description(desc=f'[Validation] Epoch {epoch + 1}/{epochs}')
                    val_avg_loss,total_avg_acc,TP,TN,FP,FN, epoch_preds, epoch_labels= self.validate(val_loader, pbar=pbar_val,is_test=False)

                    val_loss_epochs.append(val_avg_loss)
                    val_acc.append(total_avg_acc)

                    all_preds.extend(epoch_preds)
                    all_labels.extend(epoch_labels)

                    # fold_all_preds.append(all_preds)
                    # fold_all_labels.append(all_labels)

                    #save the best model
                    if total_avg_acc > best_fold_acc:
                        os.makedirs(best_fold_model_path,exist_ok=True)

                        for file_name in os.listdir(best_fold_model_path):
                            if file_name.startswith("best_model_fold_") and file_name.endswith(f"_{band}.pth"):
                                os.remove(os.path.join(best_fold_model_path, file_name))

                        best_fold_acc = total_avg_acc
                        best_fold_model_save_path = f'best_model_fold_{fold + 1}_{band}.pth'
                        torch.save(self.model.state_dict(), os.path.join(best_fold_model_path,best_fold_model_save_path))
                        
                        fold_all_preds.append(epoch_preds)
                        fold_all_labels.append(epoch_labels)
                        # all_preds.extend(epoch_preds)
                        # all_labels.extend(epoch_labels)
                        

                    # Early Stopping
                    if val_avg_loss < best_val_loss:
                        best_val_loss = val_avg_loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= self.early_stopping_patience:
                        self.end_epoch=epoch+1
                        print(f'Fold {fold+1} early stops at epoch {self.end_epoch}\n')
                        print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN,'\n')
                        
                        break

            pbar_train.close()
            fold_results.append(best_fold_acc)
            if val_loader is not None:
                pbar_val.close()

            # fold_all_preds.append(all_preds)
            # fold_all_labels.append(all_labels)

            train_loss_epochs = torch.tensor(train_loss_epochs).numpy()
            val_loss_epochs = torch.tensor(val_loss_epochs).numpy()

            if permutation_test == True:
                continue

            plt.figure()
            plt.plot(list(range(1, self.end_epoch+1)), train_loss_epochs, label='train')
            if val_loader is not None:
                plt.plot(list(range(1, self.end_epoch+1)), val_loss_epochs, label='validation')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            if permutation_test == True:
                plt.title(f'{fold+1}_{band}_loss(permutation)')
            else:
                plt.title(f'{fold+1}_{band}_loss')
            plt.show()
            plt.close('all')

            if val_loader is not None:
                train_acc=torch.tensor(train_acc).numpy()
                val_acc=torch.tensor(val_acc).numpy()
                plt.figure()
                plt.plot(list(range(1, self.end_epoch+1)), train_acc, label='train')
                plt.plot(list(range(1, self.end_epoch+1)), val_acc, label='validation')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('ACC')
                if permutation_test == True:
                    plt.title(f'{fold+1}_{band}_acc(permutation)')
                else:
                    plt.title(f'{fold+1}_{band}_acc')
                plt.show()
                plt.close('all')

            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            if permutation_test == True:
                plt.title(f'Receiver Operating Fold_{fold+1}_{band}(permutation)')
            else:
                plt.title(f'Receiver Operating Fold_{fold+1}_{band}')
            plt.legend(loc="lower right")
            plt.show()

        fold_results = [result.cpu().numpy() for result in fold_results]
        print(f'Cross-Validation Accuracy: {np.mean(fold_results)} ± {np.std(fold_results)}')

        return fold_results, fold_all_labels, fold_all_preds

    def generate_cam(self, features, weights,block):
        _, nc, h, w = features.shape
        cam = torch.zeros((h, w), dtype=torch.float32).to(self.device)

        for i in range(nc):
            cam += weights[i].item() * features[0, i, :, :]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

    # Interpolate CAM to the size of the original image
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(block.size(2), block.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        return cam
        
    def validate(self, data_loader, *, pbar=None,is_test=False) -> float:
        sigmoid = nn.Sigmoid()
        """
        :param pbar: when pbar is specified, do not print average loss
        :return:
        """       
        torch.cuda.empty_cache()

        metrics_acc = {}
        loss_acc = 0
        size_acc = 0
        total_acc=0
        is_need_log = (pbar is None)  

        # Initialize TP, TN, FP, FN
        TP, TN, FP, FN = 0, 0, 0, 0  

        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            if pbar is None:
                pbar = tqdm(total=len(data_loader), desc=f'[Validation]', unit='block')

            
            for batch in data_loader:
                self.model.eval()

                block, gt = batch
                block = self.to_device(block)
                gt = self.to_device(gt)

                # forward
                pred= self.model(block)  # [B, 1]
                
                if is_test:
                    features = self.model.get_features(block)
                    weights = self.model.get_cam_weights()

                    # 生成CAM
                    cam = self.generate_cam(features, weights,block)

                    # Plot original image and CAM
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(block[0].permute(1, 2, 0).cpu().numpy(),cmap='coolwarm')
                    plt.title('Original Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(cam.cpu().numpy(), cmap='jet')
                    plt.title('CAM')
                    plt.show()

                    print(pred)
                    print(gt)
                    print(sigmoid(pred))
                    
                    pred_mask_num = 0 if sigmoid(pred).item() < 0.5 else 1
                    print(pred_mask_num)
                    print('Prediction is correct' if pred_mask_num == gt.item() else 'Prediction is incorrect')


                # forward
                pred_mask=sigmoid(pred)

                pred_mask[pred_mask>0.5]=1
                pred_mask[pred_mask<=0.5]=0

                all_preds.extend(pred_mask.cpu().numpy())
                all_labels.extend(batch[1].to(torch.int).cpu().numpy())

                acc=torch.sum(pred_mask==batch[1].to(torch.int).to(self.device))

                TP += ((pred_mask == 1) & (batch[1].to(self.device) == 1)).sum().item()
                TN += ((pred_mask == 0) & (batch[1].to(self.device) == 0)).sum().item()
                FP += ((pred_mask == 1) & (batch[1].to(self.device) == 0)).sum().item()
                FN += ((pred_mask == 0) & (batch[1].to(self.device) == 1)).sum().item()


                step_dict = self._step(batch)
                batch_size = step_dict['batch_size']
                loss = step_dict['loss']
                acc=step_dict['acc']
                # print(acc)
                loss_value = loss.item()

                # aggregate metrics
                metrics_acc = self._aggregate_metrics(metrics_acc, step_dict)

                # update information
                loss_acc += loss_value
                size_acc += batch_size
                total_acc+=acc
                pbar.update(batch_size)
                pbar.set_postfix(loss=loss_value)


        val_avg_loss = loss_acc / size_acc
        total_avg_acc = total_acc/size_acc
        if is_test:
            print(total_avg_acc)
        pbar.set_postfix(val_avg_loss=val_avg_loss)
        if is_need_log:
            pbar.close()  # destroy newly created pbar
            print('=' * 30 + ' Measurements ' + '=' * 30)
            for k, v in metrics_acc.items():
                print(f"[{k}] {v / size_acc}")
        else:
            return val_avg_loss,total_avg_acc,TP,TN,FP,FN,all_preds, all_labels
        

    def _step(self, batch) -> dict:
        block, gt = batch

        block = self.to_device(block)  # [B, C=1, H, W]
        gt = self.to_device(gt)  # [B, C=1, H, W]
        
        B= block.shape[0]

        pred= self.model(block)  # [B, C=1, H, W]
        gt = gt.view(B,1)
        loss = self.criterion(pred, gt.to(torch.float32))

        threshold=0.5
        pred_mask=torch.sigmoid(pred)
        pred_mask[pred_mask>threshold]=1
        pred_mask[pred_mask<=threshold]=0
        acc=torch.sum(pred_mask==gt.to(torch.int))

        step_dict = {
            'loss': loss,
            'batch_size': B,
            'acc':acc
        }

        return step_dict




