<!--
 * @Author: pingping yang
 * @Date: 2024-08-23 06:27:29
 * @Description: 
-->
# Experiment information
## Data
单被试 单电极 经过stft的Ecog数据

数据形状是：`(n_samples, n_frequencies, n_timesteps)`
其中 n_samples 是n_block X n_words X n_read/n_cue  一个block一般是120个sample，因为是20个词，每个词看3遍读3遍，所以cue和read各60个，HS69有6个block，就有120x6=720个sample。

其中n_frequencies是取决于采样率，在该试验中为500HZ，窗函数长度为2*500，故n_frequencies=501

其中n_timesteps=375，在实验中每个sample取0.75s的数据，0.75*500=375
## 第一阶段Goal：
总体目标就是研究**不同频段不同脑区在不同任务上的激活**（现有研究大多基于high-gamma，我们可以尝试研究是否其他频段在语音处理上也可以提供帮助，以及其他频段在哪些语音特征上有作用）
1. `accuracy mapping`——将confused matrix绘制在脑表面
	- 目前所做的二分类在单电极上有两个维度（task，freq_band），对不同频段做二分类区分不同的任务，根据predicted和ground truth得到confused matrix，然后计算得到某个频段是否在某一类分类上做的很好或者很差，那说明这个频段在某个任务上是激活的或者不激活的。
2. `contribution`——研究不同频段对不同任务的贡献度
	- 用不同频段数据做binary classification（随机森林/resnet50）
	- 用全频段做binary classification，在exclude 1 band做binary classification

---
## 2024/8/19
和老师讨论的结果：
问题所在：
1. 现在的输入数据存在冗余，导致模型过拟合。
2. 没有搞清楚数据的形状，数据的特征，数据长什么样子

解决办法：
1. 先搞清楚数据是什么，数据每一个维度的大小是怎么得到的
2. 画出每一个freqband的形状
3. 可以使用不同的输入特征，
	1. 在freq维度上做average，取出不同freq band之后，将这一段先做normalization，再将这一段数据平均得到的结果作为分类器的输入。
	2. 在时间维度上做average，使用滑动窗口，这个时候就不管特征在哪个band，输入的都是所有的freq band
4. 计算R，可以计算不同band之间的R，比如用皮尔逊相关系数计算

others：
我们所做的东西，不是丢一堆数据让模型去学，而是观察数据，再考虑怎么样设计实验数据，然后再输入给模型。得到了结果要考虑为什么是这样的结果，这样的结果反映了什么样的特征，然后下一步怎么研究这个特征或者怎么验证，再进行下一步实验。

---
可以从以下几个方面调整
- model：
  - 除了SVM，也可以尝试其他分类器，如decision tree， random forest
  - 使用深度学习模型，如resnet，线性分类器，rnn等
- baseline
  - 统一在第一个block的(onset - freq * 0.75, onset)
  - 在每个sample前的(onset - freq * 0.75, onset)
  - cue在第一个cue前的(onset - freq * 0.75, onset)作为所有cue的baseline，read在第一个read前的(onset - freq * 0.75, onset)作为所有read的baseline
- Normalise or z-score
  - Normalization：通常指将数据缩放到一个特定的范围，比如0到1。这种方法保持了数据的相对关系，但改变了数据的绝对值范围。适用于特征需要在特定范围内进行比较时，例如当算法对特征的尺度很敏感时。
  - Z-score Standardization：通过减去均值并除以标准差，将数据转换为均值为0、标准差为1的标准正态分布。它消除了数据的均值和标准差影响，使得数据的分布更加一致，适用于数据分布偏差较大的情况。
- average dimension
  - 在frequency dimension进行平均 ：关注的是不同频率成分的差异，减少频率通道的数量，得到的数据（n_samples x n_timesteps）
  - 在time steps dimension进行平均：关注时间上的变化模式，可以总结时间序列中的整体特征，得到的数据（n_samples x n_frequencies）
- 数据降维
  - 使用pca对数据降维，减少不必要的开销以及减少噪声
- combination
  - 合并不同频段的数据观察分类结果
  - 每次加入一个频段观察分类结果

---
经历了两周的各种尝试和各种bug，终于得到的合理的分类结果！！！

## 第一阶段Results
### Analysis Data and model details
HS69  electrode74  samplingrate=500
#### original Ecog signal
![alt text](image.png)![alt text](image-1.png)

#### stft data
![alt text](image-2.png)

#### Pearson correlation matrix between frequency bands
![alt text](image-91.png)

`PCA(n_components=2)`

![alt text](image-92.png)

#### Pearson correlation matrix between samples
![alt text](image-3.png)

#### svm model
实验全部使用svm分类器，pca参数0.95，输入的数据形状为（n_samples, n_features）
```python
class SVMBinClassifier(nn.Module):
    def __init__(self):
        # tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1], 'gamma': [0.01]}]
        # self.model = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=1)
        self.model = SVC(C=0.1, kernel='sigmoid')
        self.pca = PCA(n_components=0.95)
    
    def train(self, x_train, y_train):
        X_train_pca = self.pca.fit_transform(x_train)
        self.model.fit(X_train_pca, y_train)
        plt_learning_curve(self.model, X_train_pca, y_train, cv=5)
        # best_params = self.model.best_params_
        # return best_params
    
    def predict(self, x_test):
        # X_test_pca = self.pca.transform(x_test)
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        X_test_pca = self.pca.transform(X_test)
        y_pred = self.predict(X_test_pca)
        # y_pred = cross_val_predict(self.model, X=X_test_pca, y=y_test, cv=5)
        return y_pred
```

### Accuracy
所有的z-score和average都是根据baseline求得的，cue和read各有一个baseline，都是每一个block的cue/read第一个onset - freq * 0.75到onset
- average at frequency dimension（after z-score or normalise）
- average at timesteps dimension（after z-score or normalise）
- z-score
- normalise

```python
for i in range(len_cue):
  filter_cue, filter_base_cue = self.excluding(elec_cue[i], baseline_cue[0], freq, band)
  exclu_cue.append(filter_cue)
  base_cue.append(filter_base_cue)
for j in range(len_read):
  filter_read, filter_base_read = self.excluding(elec_read[j], baseline_read[0], freq, band)
  exclu_read.append(filter_read)
  base_read.append(filter_base_read)
# normalise
X_cue_norm = (X_cue - X_base_cue) / X_base_cue
X_read_norm = (X_read - X_base_read) / X_base_read
# Compute the mean across the frequency dimension (axis=1)
if avg == 'avgFreq':
    X_cue_mean = np.mean(X_cue_norm, axis=1)
    X_read_mean = np.mean(X_read_norm, axis=1)
    X=np.concatenate((X_cue_mean, X_read_mean),axis=0)

# Compute the mean across the timesteps dimension (axis=1)
elif avg == 'avgTime':
    X_cue_mean = np.mean(X_cue_norm, axis=2)
    X_read_mean = np.mean(X_read_norm, axis=2)
    X=np.concatenate((X_cue_mean, X_read_mean),axis=0)

elif avg == 'norm':
    # X_cue_norm = (X_cue - X_base_cue) / X_base_cue
    # X_read_norm = (X_read - X_base_read) / X_base_read
    X=np.concatenate((X_cue_norm, X_read_norm),axis=0)
    X = X.reshape(-1, X.shape[1]*X.shape[2])

elif avg == 'z-score':
    X_cue_norm = self.z_score_standardize(X_cue, X_base_cue)
    X_read_norm = self.z_score_standardize(X_read, X_base_read)
    X=np.concatenate((X_cue_norm, X_read_norm),axis=0)
    X = X.reshape(-1, X.shape[1]*X.shape[2])
```

#### `normalization + average at frequency dimension`
- 准确率条形图

  ![alt text](image-54.png)
- confusion matrix
![alt text](image-55.png)

#### `normalization + average at timesteps dimension`
- 准确率条形图
  
  ![alt text](image-56.png)
- confusion matrix
![alt text](image-57.png)

#### `normalization`
- 准确率条形图
  
  ![alt text](image-58.png)
- confusion matrix
![alt text](image-59.png)

#### `z-score`
- 准确率条形图
  
  ![alt text](image-60.png)
- confusion matrix

  ![alt text](image-61.png)

#### permutation(shuffle train_y_label)
- 准确率条形图
  
  ![alt text](image-66.png)

### Contribution
所有的z-score和average都是根据baseline求得的，cue和read各有一个baseline，都是第一个block的第一个onset - freq * 0.75到onset，每次去除一个frequency band，使用其余frequency band进行分类
- average at frequency dimension（after z-score or normalise）
- average at timesteps dimension（after z-score or normalise）
- z-score
- normalise

#### `normalization + average at frequency dimension`
- allBands accuracy = 0.85
  
  ![alt text](image-71.png)
- 准确率条形图
  
  ![alt text](image-68.png)
- confusion matrix
![alt text](image-69.png)
- contribution
  
  ![alt text](image-70.png)

#### `normalization + average at timesteps dimension`
- allBands accuracy = 0.96
  
  ![alt text](image-42.png)
- 准确率条形图
  
  ![alt text](image-43.png)
- confusion matrix

  ![alt text](image-44.png)
- contribution

  ![alt text](image-22.png)

#### `normalization`
- allBands accuracy = 1

  ![alt text](image-46.png)
- 准确率条形图

  ![alt text](image-47.png)
- confusion matrix

  ![alt text](image-48.png)
- contribution

  ![alt text](image-49.png)

#### `z-score`
- allBands accuracy = 1

  ![alt text](image-50.png)
- 准确率条形图

  ![alt text](image-51.png)
- confusion matrix

  ![alt text](image-52.png)
- contribution

  ![alt text](image-53.png)

#### permutation(shuffle train_y_label) - normalization
- allBands accuracy = 0.425

  ![alt text](image-65.png)
- 准确率条形图

  ![alt text](image-63.png)
- contribution

  ![alt text](image-64.png)


### Combined one frequency every time - accuracy
#### `average at frequency dimension`
- 准确率条形图

  ![alt text](image-74.png)

#### `average at timesteps dimension`
- 准确率条形图

  ![alt text](image-75.png)

#### `normalisation`
- 准确率条形图

  ![alt text](image-73.png)

#### `z-score`
- 准确率条形图

  ![alt text](image-76.png)


### Combined two frequency - accuracy
#### `average at frequency dimension`
- 准确率条形图

  ![alt text](image-78.png)

#### `average at timesteps dimension`
- 准确率条形图

  ![alt text](image-79.png)

#### `normalisation`
- 准确率条形图

  ![alt text](image-77.png)

#### `z-score`
- 准确率条形图

  ![alt text](image-80.png)

## 第一阶段mapping
### Accuracy mapping
最终选择数据集的处理方式为normalization + average at frequency
1. 使用svm计算HS69的所有电极在不同band的分类准确率并保存y_true和y_pred为npy文件
  ```python
   # 所有电极在不同频段的acc
  idx_elec = list(range(1, 256))
  for elec in idx_elec:
    path_elec = f'/public/DATA/overt_reading/dataset_/HS{HS}/{freq}/{elec}'
    num_samples = len(os.listdir(path_elec))
    y_save_path = f'/root/pp/covert-reading/Ecog_pretrain/accuracy_results_svm_avgfreq/HS{HS}/{freq}/{elec}'
    fig_save_path = f'/root/pp/covert-reading/Ecog_pretrain/accuracy_results_svm_avgfreq/HS{HS}/{freq}/{elec}/figs'
    os.makedirs(y_save_path, exist_ok=True)
    os.makedirs(fig_save_path, exist_ok=True)
    for band in band_list:
      print(f'Calculate for HS{HS} elec{elec} band{band}')
      data_loader = SVMDataset(HS, path_elec, freq, elec, num_samples, band, exclude=False, avg='avgFreq')
      data, labels = data_loader.get_data_labels()
      print(f'data_shape: {data.shape}')
      X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=1/6, random_state=42)
      # svm = DecisionTreeBinClassifier()
      svm = SVMBinClassifier()
      svm.train(x_train=X_train, y_train=y_train)

      y_pred = svm.evaluate(X_test=X_test, y_test=y_test)
      band_acc = accuracy_score(y_test, y_pred)
      print(f"Accuracy on test set band_{band}: {band_acc}")

      np.save(os.path.join(y_save_path, f'{band}_y_pred.npy'), y_pred)
      np.save(os.path.join(y_save_path, f'{band}_y_true.npy'), y_test)
  ```
2. 根据保存的npy文件计算准确率并存为字典acc_dic 字典的键值是elec，对应的value是还是一个字典，子字典的key是band， value是acc，所以电极对应band的acc为 acc_dic[elec][band]
  ```python
  def cal_all_elec_acc(rootPath, HS, freq):
    acc_dic = {}
    band_list = get_all_band()
    idx_elec = list(range(1, 256))
    
    for elecidx in idx_elec:
        band_acc = {}
        for band in band_list:
            # y_true_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_true.npy')
            # y_pred_path = os.path.join(rootPath, f'HS{HS}/{freq}/{elecidx}/{band}_y_pred.npy')
            acc = cal_acc_band1_from_y(rootPath, band)
            band_acc[band] = acc
        acc_dic[elecidx] = band_acc
    return acc_dic
  ```
3. 将每个电极对应的所有band的分类准确率叠加映射到一个脑模板上，每个band使用不同的形状，acc用颜色映射表示
  ![alt text](image-83.png)
4. 只显示acc>0.9的频段的电极
   >Total 72 electrodes accuracy with else1 > 0.9
Total 65 electrodes accuracy with delta > 0.9
Total 64 electrodes accuracy with theta > 0.9
Total 67 electrodes accuracy with alpha > 0.9
Total 65 electrodes accuracy with beta > 0.9
Total 69 electrodes accuracy with gamma > 0.9
Total 69 electrodes accuracy with high gamma > 0.9
Total 73 electrodes accuracy with else2 > 0.9
   ![alt text](image-82.png)
5. 每个频段单独显示在一张子图上
   ![alt text](image-84.png)
6. 只显示acc>0.9的频段的电极
   ![alt text](image-85.png)
7. 绘制confusion matrix到脑表面
  ![alt text](image-86.png)
8. 绘制 recall 到脑表面
   ![alt text](image-87.png)
9.  绘制 specificity 到脑表面
10. ![alt text](image-88.png)
11. 统计每个电极所有频段的平均准确率
    ![alt text](image-93.png)
12. 统计每个band的平均准确率
    
    ![alt text](image-90.png)
### Contribution mapping
1. 计算all_bands的分类准确率，保存为npy文件
  ```python
  # 计算all_bands acc
  for elec in idx_elec:
      path_elec = f'/public/DATA/overt_reading/dataset_/HS{HS}/{freq}/{elec}'
      num_samples = len(os.listdir(path_elec))
      y_save_path = f'/root/pp/covert-reading/Ecog_pretrain/accuracy_results_svm_avgfreq/HS{HS}/{freq}/{elec}'
      os.makedirs(y_save_path, exist_ok=True)

      data_loader = SVMDataset(HS, path_elec, freq, elec, num_samples, avg='avgFreq')
      # data_loader = CombineBandDataset(HS, path_elec, freq, elec, num_samples, band_list, avg='avgFreq')
      data, labels = data_loader.get_data_labels()
      # print(f'data_shape: {data.shape}')

      X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=1/6, random_state=42)
      # print(f'x_train shape：{X_train.shape}, x_test shape：{X_test.shape}, y_train shape：{y_train.shape}, y_test shape：{y_test.shape}')# (600, 375), x_test shape：(120, 375), y_train shape：(600,), y_test shape：(120,)

      svm = SVMBinClassifier()
      # svm = DecisionTreeBinClassifier()
      svm.train(x_train=X_train, y_train=y_train)

      y_pred = svm.evaluate(X_test=X_test, y_test=y_test)
      all_band_acc = accuracy_score(y_test, y_pred)

      np.save(os.path.join(y_save_path, f'allbands_y_pred.npy'), y_pred)
      np.save(os.path.join(y_save_path, f'allbands_y_true.npy'), y_test)

      print(f"Accuracy on test set - elec{elec}: {all_band_acc}")
  ```
1. 计算每个band的分类准确率，并根据保存的all_bands准确率计算contribution
  ```python
  # 所有电极在不同频段的contribution
  for elec in idx_elec:
      path_elec = f'/public/DATA/overt_reading/dataset_/HS{HS}/{freq}/{elec}'
      num_samples = len(os.listdir(path_elec))
      y_save_path = f'/root/pp/covert-reading/Ecog_pretrain/accuracy_results_svm_avgfreq/HS{HS}/{freq}/{elec}'
      fig_save_path = f'/root/pp/covert-reading/Ecog_pretrain/accuracy_results_svm_avgfreq/HS{HS}/{freq}/{elec}/figs'
      contribution_save_path = f'/root/pp/covert-reading/Ecog_pretrain/contribution_results_svm_avgfreq/HS{HS}/{freq}/{elec}'
      os.makedirs(y_save_path, exist_ok=True)
      os.makedirs(fig_save_path, exist_ok=True)
      os.makedirs(contribution_save_path, exist_ok=True)
      baseline_accuracy = cal_acc_band1_from_y(y_save_path, 'allbands')
      contributions_list = []
      for band in band_list:
          data_loader = SVMDataset(HS, path_elec, freq, elec, num_samples, band, exclude=True, avg='avgFreq')
          data, labels = data_loader.get_data_labels()
          print(f'data_shape: {data.shape}')
          X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=1/6, random_state=42)
          # svm = DecisionTreeBinClassifier()
          svm = SVMBinClassifier()
          svm.train(x_train=X_train, y_train=y_train)

          y_pred = svm.evaluate(X_test=X_test, y_test=y_test)
          band_acc = accuracy_score(y_test, y_pred)
          print(f"Accuracy on test set without band_{band}: {band_acc}")

          np.save(os.path.join(y_save_path, f'{band}_y_pred.npy'), y_pred)
          np.save(os.path.join(y_save_path, f'{band}_y_true.npy'), y_test)
          # plt confusion matrix
          plt_confusion_matric(y_test, y_pred, HS, elec, freq, f'without_{band}')

          band_acc = cal_acc_band1_from_y(y_save_path, band)
          contribution = baseline_accuracy - band_acc
          print(f'band: {band}, contribution_acc:{contribution}')
          contributions_list.append(contribution)
      np.save(os.path.join(contribution_save_path, f'contributions.npy'), contributions_list)
  ```
  ```python
  contribution_dic = {}
  for elec in idx_elec:
      contribution_save_path = f'/root/pp/covert-reading/Ecog_pretrain/contribution_results_svm_avgfreq/HS{HS}/{freq}/{elec}'
      contribution_list = np.load(os.path.join(contribution_save_path, f'contributions.npy'))
      contribution_dic[elec] = contribution_list
  ```
  1. 将每个电极不同band的contribution以扇形图的方式画到脑表面
  ![alt text](image-89.png)


## 2024/9/9
开会讨论结果：
- 实验选择的baseline不科学，引入了额外的信息，统一cue和read的baseline为每一个block第一个cue之前的0.75s
- 特征太多，选择的频率点过于密集，可以考虑间隔选择一些点
- 找出数据结果很差的原因，从可视化原始特征开始，做t-test，计算cue和read data的显著性点并可视化

## t-test
1. baseline：cue前的(0, 0.75), 统一cue和read为一个baseline
2. average：分别将cue和read的samples平均，得到一个shape为（n_timesteps, n_frequencies）的数据
3. plot：分别绘制cue和read的频谱图
4. t-test：逐点计算cue和read的显著性
### original data
将cue和read的所有samples分别平均，绘制原始数据的频谱图
![alt text](image-94.png)

### z-score data
将cue和read的所有samples先z-score再进行平均
![alt text](image-95.png)

### t-test and plot significant area
```python
# t_test
n_frequency, n_times = data_cue_norm[0].shape

print(data_cue_norm.shape, data_read_norm.shape)
print(np.isnan(data_cue_norm).sum(), np.isnan(data_read_norm).sum())
from scipy import stats
import numpy as np

# 存储 p 值矩阵
p_values = np.zeros((n_frequency, n_times))
f_values = np.zeros((n_frequency, n_times))
for f in range(n_frequency):
    for t in range(n_times):
        cue_vals = data_cue_norm[:, f, t]
        read_vals = data_read_norm[:, f, t]
        
        if np.std(cue_vals) > 0 and np.std(read_vals) > 0:
            f_val, p_val = stats.ttest_ind(cue_vals, read_vals, equal_var=False)
            p_values[f, t] = p_val
            f_values[f, t] = f_val
        else:
            p_values[f, t] = np.nan
            f_values[f, t] = np.nan
```

### t-test on 256 electrodes

- 显示频率时间点显著区域的电极总和
  
  1）首先计算每个点的所有显著电极数量的总和并可视化
  combined_significant_points = np.sum(significant_points_all_electrodes, axis=0)
  ![alt text](image-98.png)

  2）可视化至少有一个显著电极的点

  ![alt text](image-102.png)



- 显示每个电极的显著点的百分比

  1）计算第90百分位的阈值，并可视化所有电极的显著点百分比，其中超出阈值的电极的占比为 10.16%，超出阈值的所有电极的平均显著点百分比为 10.62%。
  ![alt text](image-100.png)

  2）可视化超过阈值的电极的显著点百分比

  ![alt text](image-101.png)

- 在部分电极的 average spectrum 上可视化显著点以及圈出显著区域

电极23是不显著电极

![alt text](image-110.png)
![alt text](image-104.png)

电极45是不显著电极

![alt text](image-111.png)
![alt text](image-112.png)

电极65是不显著电极
![alt text](image-113.png)
![alt text](image-114.png)

电极74是cue trail上的显著电极

![alt text](image-115.png)
![alt text](image-116.png)

![alt text](image-117.png)
![alt text](image-118.png)

![alt text](image-119.png)
![alt text](image-120.png)

![alt text](image-121.png)
![alt text](image-122.png)

电极138是read trail上的显著电极

![alt text](image-123.png)
![alt text](image-124.png)

- 将每个电极average的p-value mapping到标准脑
```python
avg_elec_pval = {}
for elec in range(256):
    avg_elec_pval[elec] = np.mean(p_values_all_electrodes[elec])

norm = Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('viridis')  # 使用统一的颜色映射

x_coords, y_coords, p_values = [], [], []  
for elec in range(256):
    pval = avg_elec_pval[elec]
    x_coords.append(xy[elec][0])
    y_coords.append(xy[elec][1])
    p_values.append(pval)
    print(f'Avergae p-value for electrode {elec}: {pval}')

scatter = ax.scatter(
    x_coords, y_coords,
    c=p_values,
    cmap=cmap,
    norm=norm,  # 统一的标准化范围
    alpha=0.5,
    s=50,
)

```
![alt text](image-109.png)

## 2024/9/13
讨论结果
- 从以上t-test看不出数据问题，再回到上一层，画原始erp数据
- 绘制high-gamma的erp
- 绘制低频的erp
- 可以将两类trails的erp做一个减法
- 可以做一个低通滤波
加一个
-分频段绘制t-test结果

## 2024/9/19
修改baseline，根据振杰师兄提供的思路，使用每个block每个cue onset前的0.2s拼接在一起作为baseline，计算z-score
![alt text](image-125.png)
```python
def z_score_standardize(X, X_base):
    """计算 z-score 标准化"""
    # 计算拼接后的均值和标准差
    mean_baseline = np.mean(baseline, axis=(0, 2))  # 在样本和时间维度上计算均值
    std_baseline = np.std(baseline, axis=(0, 2))    # 在样本和时间维度上计算标准差

    # 扩展均值和标准差的维度以便与 X 进行广播
    mean_baseline = mean_baseline[np.newaxis, :, np.newaxis]  # 形状变为 (1, n_frequencies, 1)
    std_baseline = std_baseline[np.newaxis, :, np.newaxis]    # 形状变为 (1, n_frequencies, 1)

    # 进行 z-score 标准化
    return (X - mean_baseline) / std_baseline
```
```python
for num in range(num_samples): # num为块的个数
    cue_path = os.path.join(path_elec, f'{num}_data_block_cue.npy')
    read_path = os.path.join(path_elec, f'{num}_data_block_read.npy')
    baseline_path = os.path.join(path_elec, f'{num}_baseline_block_cue.npy')
    # print(cue_path)
    if os.path.exists(cue_path) and os.path.exists(read_path):
        elec_cue = np.load(cue_path) # (n_task, n_freq, n_timePoint) (60, 501, 375)
        elec_read = np.load(read_path)
        elec_base = np.load(baseline_path)

        data_cue.append(elec_cue)
        data_read.append(elec_read)
        baseline_data.append(elec_base[:, :, :100])# 100=500*0.2


  data_cue=np.abs(np.vstack(data_cue))
  data_read=np.abs(np.vstack(data_read))
  baseline_data=np.abs(np.vstack(baseline_data))
  # calculate z-score
  data_cue_norm = z_score_standardize(data_cue, baseline_data)
  data_read_norm = z_score_standardize(data_read, baseline_data)
```
### Results - allbands（average spectrogram & t-test）
- elec 25

  ![alt text](image-126.png)
  ![alt text](image-127.png)

- elec 74 cue trail的显著电极

  ![alt text](image-128.png)
  ![alt text](image-129.png)

- elec 138 read trail的显著电极

  ![alt text](image-130.png)
  ![alt text](image-131.png)

- elec 111

  ![alt text](image-132.png)
  ![alt text](image-133.png)

## Results - spectrogram for each words
将单个电极的不同词的频谱图绘制出来看是否有显著差异
![alt text](image-158.png)
![alt text](image-159.png)
![alt text](image-160.png)
![alt text](image-161.png)
![alt text](image-162.png)
![alt text](image-163.png)

### Results - split frequency bands（average spectrogram）
- elec 74

  ![alt text](image-134.png)
  ![alt text](image-135.png)
  ![alt text](image-136.png)
  ![alt text](image-137.png)
  ![alt text](image-138.png)
  ![alt text](image-139.png)
  ![alt text](image-140.png)
  ![alt text](image-141.png)

### Result - accuracy for each band
![alt text](image-156.png)
![alt text](image-157.png)
### 结果思考
>从上面的结果来看，cue和read的数据之间还是存在差异的，但是分类的结果很差，有没有可能虽然两个类别的差距很大，但是每个类别的类内的差异也很大，不同的词可能产生的erp不同，导致svm无法产生较为合理的决策边界呢？————可以先不平均所有的trail，平均相同word的不同的trail绘制spectrogram或者erp？

## Results - erp analysis
### cue and read data erps
![alt text](image-149.png)
### cue - read data erps
![alt text](image-150.png)
### high gamma erps
![alt text](image-151.png)
![alt text](image-164.png)
### low frequency erps
![alt text](image-152.png)
### z-score cue and read data
![alt text](image-153.png)

### erps analysis for each words
不确定是否相同的任务上，不同的词之间也存在着很大的差异，所以绘制相同任务的词的erps
选择的**word_list = ['树叶','数页','shù yè','对十','绿草']**

数据处理是，对于word_list中的每一个词，切割出每个词对应的cue data和read data的所有trail，得到的数据是(n_trails, n_electrodes, n_timesteps)， 将每一个词的所有trail平均，得到(n_electrodes, n_timesteps)，然后选择其中几个电极绘制结果

![alt text](image-154.png)

### z-score erps analysis for each words
![alt text](image-155.png)

## 2024/9/23
和老师的讨论结果：有可能确实variance比较大，现在就是针对每个频谱，画出来每个电极的平均所有trail频谱的振幅，也要画出std

绘制stft得到的频谱图的各个频段的平均值，在计算平均之前，先对所有trail做一个z-score
### Results - Average Amplitude with z-score
![alt text](image-165.png)
![alt text](image-166.png)
![alt text](image-173.png)
![alt text](image-167.png)
![alt text](image-174.png)
![alt text](image-175.png)
![alt text](image-168.png)
![alt text](image-176.png)
![alt text](image-169.png)


### REsults - Avergae at different words
![alt text](image-188.png)
![alt text](image-189.png)
![alt text](image-190.png)

### Results - SE(std / 根号n_trail) after z-score
![alt text](image-191.png)
![alt text](image-193.png)
![alt text](image-192.png)
![alt text](image-194.png)
![alt text](image-195.png)
![alt text](image-196.png)
![alt text](image-197.png)
![alt text](image-198.png)

### Results - Avg spectrogram + SE(z-score for each block)
之前的z-score是所有的block一起计算的，现在的z-score是单独计算每一个block，每个block的baseline都是取每一个cue_onset的前0.2s数据拼接在一起得到的。
- allbands(0-501HZ) mean amplitude for each electrode
![alt text](image-257.png)
- else1(0-1HZ) band mean amplitude for each electrode
![alt text](image-258.png)
- delta(1-4HZ) band mean amplitude for each electrode
![alt text](image-259.png)
- theta(4-8HZ) band mean amplitude for each electrode
![alt text](image-260.png)
- alpha(8-12HZ) band mean amplitude for each electrode
![alt text](image-261.png)
- beta(12-30HZ) band mean amplitude for each electrode
![alt text](image-262.png)
- gamma(30-70HZ) band mean amplitude for each electrode
![alt text](image-263.png)
- high gamma(70-150HZ) band mean amplitude for each electrode
![alt text](image-264.png)
- else2(150-501HZ) band mean amplitude for each electrode
![alt text](image-265.png)

- thinking
1. 在不同任务开始之后出现的抑制是因为什么？尤其在低频段出现了read的抑制，高频段出现的cue的抑制。
2. SE较大，尤其是cue任务

### Binary Classification Accuracy（z-score）：

- electrode 74
![alt text](image-225.png)
![alt text](image-240.png)
![alt text](image-252.png)

- electrode 138
![alt text](image-228.png)
![a lt text](image-241.png)
![alt text](image-253.png)

- electrode 129
![alt text](image-232.png)
![alt text](image-242.png)
![alt text](image-254.png)

- electrode 215
![alt text](image-245.png)
![alt text](image-243.png)
![alt text](image-255.png)

- electrode 25

--- 


### Binary Classification Accuracy（avgFreq）：
将数据在frequency维度平均，分类器的输入shape=（n_trails， n_timesteps）=（720， 375）
- electrode 138
![alt text](image-234.png)
![alt text](image-247.png)

- electrode 74
![alt text](image-237.png)
![alt text](image-239.png)

- electrode 25
![alt text](image-249.png)
![alt text](image-251.png)

一些思考：
从上面的结果来看，不同频段在两类任务上的分类准确率是有差别的，不同频段的SE也有差距，并且全频段SE是最大的，分类效果也最差
1. **频段的差异性**
不同频段（如高伽马、β波段等）可能与特定任务相关。**不同任务激活不同频段**
（1）视觉任务：通常涉及到视觉处理区域（如枕叶和颞叶），可能更依赖于较高频段（如高伽马和gamma波），这些频段与注意力、感知和信息整合有关。
（2）发音任务：主要涉及听觉处理区域（如上颞回）和运动计划区域，可能更多依赖于中低频段（如β波和gamma波），这些频段与语言处理、音素辨别等认知功能密切相关。
2. **全频段数据的影响**
使用全频段数据时，模型的准确率接近随机水平，说明（1）相互干扰：当多个频段的信号同时作用时，可能导致信号的混淆和干扰，影响大脑对特定任务的执行。这是因为不同频段可能传递了不同类型的信息，过多的信息会使得大脑难以专注于当前任务。
（2）分类和识别困难：在分类任务中，如果数据包含了多个频段的混合信号，分类器可能难以从中提取出有用的特征，从而降低准确性。

### Results - Avg spectrogram for each words(z-score for each block + SE)
![alt text](image-220.png)
![alt text](image-221.png)
![alt text](image-222.png)
![alt text](image-223.png)

### Results - downsampling and filter
![alt text](image-256.png)

## PCA & ICA analysis
### PCA analysis
将原始数据预处理结束后，根据[400, 500, 1000]三个采样率进行下采样，然后将所有block的数据按照电极方向拼接在一起，对所有数据做一个去平均，然后使用PCA分析。
```python
def pca_freq(stft_result, wanted_comp):
    '''
    stft_result：shape=(n_freqs, n_times)
    '''
    mean_over_time = np.mean(stft_result, axis=1, keepdims=True)
    normalized_stft = stft_result - mean_over_time

    pca = PCA(n_components=wanted_comp)
    pca_result = pca.fit_transform(10 * np.log10(abs(normalized_stft) + 1e-10))
    components = pca.components_
    ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(ratio)
    return pca_result, components, ratio, cumulative_ratio
```
### results - pca
然后绘制500HZ采样率下的pca分析结果，前100个主成分解释的累积方差和前十个主成分中不同频段所做的方差贡献
![alt text](image-266.png)
![alt text](image-267.png)
![alt text](image-268.png)

### 对不同frequency band使用PCA
使用滤波将数据分为不同frequency bands，
```python
file_path='/public/DATA/overt_reading/concat_block/'
HS = 69
freq = 500
ecog_path = f'/public/DATA/overt_reading/concat_block/concat_HS{HS}/{freq}/block.npy'
ecog_concat = np.load(ecog_path)
print(f'HS{HS} ecog_concat.shape: {ecog_concat.shape}')
result1,f1 = parallel_stft(ecog_concat,freq)
# 将stft得到的数据所有的elec拼接在一起，每一个块的stft是（401， 1250）一共256个elect
concatenated_stft_blocks = np.concatenate(result1,axis = 1)
print(f'concatenated_stft_block shape: {concatenated_stft_blocks.shape}')

# 存储不同频段的数据
filtered_signals = {}
for band, (low, high) in freq_bands.items():
    filtered_signals[band] = bandpass_filter(ecog_concat, low, high, freq)

# 查看滤波后的数据形状
for band in filtered_signals:
    print(f"{band} band filtered data shape: {filtered_signals[band].shape}")
```
### results - pca for different frequency bands
然后对500HZ采样率下的数据使用PCA，得到不同frequency bands下的主成分分布和各成分解释的累积方差
- delta
![alt text](image-269.png)
![alt text](image-270.png)

- theta
![alt text](image-271.png)
![alt text](image-280.png)

- alpha
![alt text](image-281.png)
![alt text](image-282.png)

- beta
![alt text](image-283.png)
![alt text](image-284.png)

- gamma
![alt text](image-285.png)
![alt text](image-286.png)

- high gamma
![alt text](image-287.png)
![alt text](image-288.png)

- else2
![alt text](image-289.png)
![alt text](image-290.png)
### ICA analysis for different frequency bands
```python
def ica_analysis(stft_result, n_components):
    '''
    进行ICA分析
    parameters:
        stft_result: 输入的STFT数据 shape=(n_freqs, n_times)
        n_components: 想要提取的独立成分数量
    returns:
        ica_result: ICA后的独立成分 shape=(n_components, n_times)
        mixing_matrix: 混合矩阵，表示独立成分的组合方式
    '''
    # 对每个频段的时间序列做去均值处理
    mean_over_time = np.mean(stft_result, axis=1, keepdims=True)
    centered_stft = stft_result - mean_over_time

    # 初始化ICA模型
    ica = FastICA(n_components=n_components, random_state=0)
    
    # 进行ICA分析
    ica_result = ica.fit_transform(centered_stft.T)  # shape=(n_times, n_components)
    mixing_matrix = ica.mixing_  # 获取混合矩阵
    
    return ica_result.T, mixing_matrix  # 转置回来变为 shape=(n_components, n_times)
```

### Results - ica for different frequency bands
- delta
![alt text](image-291.png)

- theta
![alt text](image-292.png)

- alpha
![alt text](image-293.png)

- beta
![alt text](image-294.png)

- gamma
![alt text](image-295.png)

- high gamma
![alt text](image-296.png)

- else2
![alt text](image-297.png)

### PCA for different task
对不同任务使用PCA分析
1. 读出对齐后的erp数据
2. 对不同的任务做时频分析
3. 对标准化后的数据分别执行 PCA
4. 比较两个任务的前几个主成分或独立成分的分布和方差贡献。

```python
# 时频分析
file_path='/public/DATA/overt_reading/concat_block/'
HS_list=[69]
freq_list = [500]
for HS in HS_list:
    for freq in freq_list:
        PATH = f'/public/DATA/overt_reading/concat_block/concat_HS{HS}/{freq}'
        cue_ecog_concat = np.load(os.path.join(PATH, 'cue_block.npy'))
        read_ecog_concat = np.load(os.path.join(PATH, 'read_block.npy'))
        cue_result, cue_f = parallel_stft(cue_ecog_concat, freq)
        read_result, read_f = parallel_stft(read_ecog_concat, freq)

        # 将stft得到的数据所有的elec拼接在一起，每一个块的stft是（501， 1250）一共256个elect
        concatenated_cue_stft_blocks = np.concatenate(cue_result, axis = 1)
        concatenated_read_stft_blocks = np.concatenate(read_result, axis = 1)
        print(f'conctenated_cue_stft_blocks shape: {concatenated_cue_stft_blocks.shape}') 
        # 主成分分析，得到所有的主成分系数以及每一个主成分解释的方差比例
        pca_result_cue,components_cue,ratio_cue,cumu_ratio_cue = pca_freq(concatenated_cue_stft_blocks,100)

        pca_result_read,components_read,ratio_read,cumu_ratio_read = pca_freq(concatenated_read_stft_blocks,100)

        plot_pca_variance_comparison(ratio_cue, ratio_read, HS, freq)
        plot_pca_cumulative_variance(cumu_ratio_cue, cumu_ratio_read, HS, freq)
        plot_pca_frequency_comparison(components_cue, components_read, freq, HS, n_components=5)
        plot_pca_scatter(pca_result_cue, pca_result_read, n_components=2)

```
### Results - pca for different task
![alt text](image-272.png)
![alt text](image-273.png)
![alt text](image-274.png)
![alt text](image-275.png)
![alt text](image-276.png)
![alt text](image-277.png)
![alt text](image-278.png)
![alt text](image-279.png)