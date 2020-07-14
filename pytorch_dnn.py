import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import torch.utils.data as Data
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.decomposition import PCA
from tqdm import tqdm
import time

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from scipy.stats import kurtosis



def add_feature(data):

    #质量
    data['jet_mass']=np.abs(data['jet_mass']) # 原始jet_mass有负数，所以取个绝对值好点

    #横向动量的大小
    data['pt']=np.sqrt(data['jet_py']*data['jet_py']+data['jet_pz']*data['jet_pz'])
    #总动量的大小
    data['p']=np.sqrt(data['jet_px']*data['jet_px']+data['jet_py']*data['jet_py']+data['jet_pz']*data['jet_pz'])


    # 物理中通常使用 赝快度，快度
    #赝快度
    data['eta']=np.arcsinh(data['jet_px']/data['pt'])
    #快度
    data['y']=1/2*np.log( (data['jet_energy']+data['jet_px'])/ (data['jet_energy']-data['jet_px']+0.00001) )

    # 赝快度，快度 的大小
    data['abs_eta']=np.abs(data['eta'])
    data['abs_y']=np.abs(data['y'])

    # 动量夹角的余弦值   分动量的大小/总动量的大小
    data['p_x'] = np.abs(data['jet_px'])/data['p']
    data['p_y'] = np.abs(data['jet_py'])/data['p']
    data['p_z'] = np.abs(data['jet_pz'])/data['p']

    # 速度
    data['vx']=data['jet_px']/data['jet_energy']
    data['vy']=data['jet_py']/data['jet_energy']
    data['vz']=data['jet_pz']/data['jet_energy']
    # 速度的大小
    data['abs_vx']=np.abs(data['jet_px'])/data['jet_energy']
    data['abs_vy']=np.abs(data['jet_py'])/data['jet_energy']
    data['abs_vz']=np.abs(data['jet_pz'])/data['jet_energy']


    # 部分特征的平方
    # 他们满足以下物理关系
    # data['jet_energy2']-data['p2']=data['jet_mass2']
    # data['p2']=data['jet_px2']+data['jet_py2']+data['jet_pz2']
    data['p2']=data['p']*data['p']
    data['jet_px2']=data['jet_px']*data['jet_px']
    data['jet_py2']=data['jet_py']*data['jet_py']
    data['jet_pz2']=data['jet_pz']*data['jet_pz']

    data['jet_energy2']=data['jet_energy']*data['jet_energy']
    data['jet_mass2']=data['jet_mass']*data['jet_mass']

    # 物理意义不清楚
    data['jet_energy_mass'] = data['jet_energy'] / (data['jet_mass'] + 0.001)
    # 质量/横向动量 （论文中找到的，意义不清楚）
    data['m_pt']=data['jet_mass']/data['pt']
    data['m_pt']=np.log(data['m_pt']+1)

    #极化角度
    data['theta']=2* np.arctan(np.exp(-data['eta']))

    # 偏转角度
    data['phi']=np.arcsin(data['jet_py']/data['pt'])

    # 与 jet 中的粒子数量 有关的特征
    data['jet_mass_mean']=np.abs(data['jet_mass']) / data['number_of_particles_in_this_jet']
    data['jet_energy_mean']=data['jet_energy'] / data['number_of_particles_in_this_jet']

    data['jet_px_mean']=data['jet_px'] / data['number_of_particles_in_this_jet']
    data['jet_py_mean']=data['jet_py'] / data['number_of_particles_in_this_jet']
    data['jet_pz_mean']=data['jet_pz'] / data['number_of_particles_in_this_jet']

    data['p_mean'] = data['pt'] / data['number_of_particles_in_this_jet']
    data['pt_mean'] = data['pt'] / data['number_of_particles_in_this_jet']

    data['eta_mean'] = data['eta'] / data['number_of_particles_in_this_jet']
    data['y_mean'] = data['y'] / data['number_of_particles_in_this_jet']

    data['abs_eta_mean'] = data['abs_eta'] / data['number_of_particles_in_this_jet']
    data['abs_y_mean'] = data['abs_y'] / data['number_of_particles_in_this_jet']

##
    data['pt_p']=np.sqrt(data['jet_py']*data['jet_py']+data['jet_px']*data['jet_px'])
    data['eta_p']=np.arcsinh(data['jet_pz']/data['pt_p'])
    data['theta_p']=2* np.arctan(np.exp(-data['eta_p']))
    data['y_p']=1/2*np.log( (data['jet_energy']+data['jet_pz'])/ (data['jet_energy']-data['jet_pz']+0.00001) )
    data['m_pt_p']=data['jet_mass']/data['pt_p']
    data['m_pt_p']=np.log(data['m_pt_p']+1)
##
    # jet number
    data['event_id_count'] = data.groupby(['event_id'])['jet_id'].transform('count')
    xx=['jet_energy','jet_mass','jet_px','jet_py','jet_pz','number_of_particles_in_this_jet','pt','p','eta','y','abs_eta',\
     'abs_y','p_x','p_y','p_z','vx','vy','vz','abs_vx','abs_vy','abs_vz','p2','jet_px2','jet_py2','jet_pz2','jet_energy2',\
     'jet_mass2','jet_energy_mass','m_pt','theta','phi','jet_mass_mean','jet_energy_mean','jet_px_mean','jet_py_mean',\
     'jet_pz_mean','p_mean','pt_mean','eta_mean','y_mean','abs_eta_mean','abs_y_mean']
    for i in tqdm(xx):
        data = data.merge(data.groupby(['event_id'], as_index=False)[i].agg({
                'event_id_{}_mean'.format(i): 'mean','event_id_{}_std'.format(i): 'std',
                'event_id_{}_sum'.format(i): 'sum','event_id_{}_median'.format(i): 'median',
                'event_id_{}_max'.format(i): 'max','event_id_{}_min'.format(i): 'min',
                }), on=['event_id'], how='left')
        data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['jet_id'].transform('count')
    return data


# normalize
def normal(dfn1, dfn2):
    if len(dfn1.columns) != len(dfn1.columns):
        print("bad")
        return
    for i in dfn1.columns:
        mi = min(dfn1[i].min(), dfn2[i].min())
        ma = max(dfn1[i].max(), dfn2[i].max())

        dfn1[i] = (dfn1[i] - mi) / (ma - mi)
        dfn2[i] = (dfn2[i] - mi) / (ma - mi)


class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.dis = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, 100),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.BatchNorm1d(50),
            nn.Linear(50, 50),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.BatchNorm1d(50),
            nn.Linear(50, 19)
        )

    def forward(self, x):
        x = self.dis(x)
        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

#         score = -val_loss
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation acc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint1.pt')
        self.val_loss_min = val_loss


def make_dataset(x, y):
    x = torch.tensor(np.array(x), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.long)
    # 返回数据集合
    return Data.TensorDataset(x, y)


def create_datasets(batch_size, shuf, train_x, train_y):
    train_loader = Data.DataLoader(
        dataset=make_dataset(train_x, train_y),  # TensorDataset类型数据集
        batch_size=batch_size,  # mini batch size
        shuffle=shuf,  # 设置随机洗牌
        num_workers=4,  # 加载数据的进程个数 cuda 好像不能设置
        pin_memory=True)
    return train_loader


def create_datasets_prediction(batch_size, shuf, train_x):
    test = torch.tensor(np.array(train_x), dtype=torch.float)
    test_loader = Data.DataLoader(
        dataset=test,  # TensorDataset类型数据集
        batch_size=batch_size,  # mini batch size
        shuffle=shuf,  # 设置随机洗牌
        num_workers=4,  # 加载数据的进程个数 cuda 好像不能设置
        pin_memory=True)

    return test_loader


def make_prediction(net, test_loader):
    with torch.no_grad():
        for i, inputs in enumerate(test_loader, 0):
            net.eval()
            inputs = inputs.cuda(non_blocking=True)
            outputs = net(inputs)
            # _, predicted = torch.max(outputs, 1)
            outputs = torch.sigmoid(outputs.cpu().detach()).numpy()

            if i == 0:
                prediction = outputs
            else:
                prediction = np.append(prediction, outputs, axis=0)

    return prediction


def train_model(net, trainloader, testloader, patience, n_epochs):
    Acc = []
    Auc = []
    fig = plt.figure(figsize=(5, 5), dpi=100)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # train
        for i, data in enumerate(trainloader, 0):
            net.train()
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # validation
        with torch.no_grad():
            for ii, data in enumerate(testloader, 0):
                net.eval()
                inputs, labels = data
                inputs = inputs.cuda(non_blocking=True)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)

                predicted = predicted.cpu().detach().numpy()
                labels = labels.numpy()

                if ii == 0:
                    all_predicted = predicted
                    all_labels = labels
                else:
                    all_predicted = np.append(all_predicted, predicted, axis=0)
                    all_labels = np.append(all_labels, labels, axis=0)
            ac = accuracy_score(all_labels, all_predicted)
            auc = roc_auc_score(pd.get_dummies(all_labels).values,
                                pd.get_dummies(all_predicted).values,
                                average='macro')  # weighted
            print('Epoch: ', epoch, '| Loss: ', running_loss, '| Accuracy: ', ac, '%', '| AUC: ', auc)

        # fig
        Acc.append(ac * 100)
        Auc.append(auc * 100)

        plt.clf()
        ax = plt.subplot(1, 1, 1)
        plt.grid(color='r', linestyle='--', linewidth=1, alpha=0.3)
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.plot(range(len(Acc)), Acc)
        plt.plot(range(len(Auc)), Auc)
        plt.savefig('Acc1.png', bbox_inches='tight', pad_inches=0.05)
        # early stop
        early_stopping(auc, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')
    net.load_state_dict(torch.load('checkpoint1.pt'))
    return net


# begin load dataset
train = pd.read_csv("./jet_simple_data/simple_train_R04_jet.csv")
train = train.sort_values('event_id').reset_index(drop=True)
dit = {21:0,1:1,4:2,5:3}
train['label'] = train['label'].map(dit)
y = train['label']
train= train.drop(['label'], axis=1)

test = pd.read_csv("./jet_simple_data/simple_test_R04_jet.csv")
test_event_id=test['event_id']




# 预处理
temp=pd.concat([train,test],ignore_index=True)
temp_len=train.shape[0]
del train
del test

# add_feature 有两套特征，add_feature2是第二套特征
temp=add_feature(temp)
# temp=add_feature2(temp)

train=temp.loc[:temp_len-1,:]
test=temp.loc[temp_len:,:]
print(train.shape,test.shape,temp.shape)
del temp
del temp_len

train= train.drop(['jet_id','event_id'], axis=1)
test = test.drop(['jet_id','event_id'], axis=1)


train=train.fillna(0)
test=test.fillna(0)

normal(train,test)
dim=train.shape[1]
print(dim)



batch_size=256
learning_rate = 0.00001

# 单折 10个种子融合
# 10 seed
seeds = [19970412, 2019 * 2 + 1024, 4096, 2048, 1024,1,2,3,4,5,6,7,8,9]
num_model_seed = 10
for model_seed in range(num_model_seed):
    print(model_seed + 1)

    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=False)
    for index, (train_index, test_index) in enumerate(skf.split(train, y)):
        print(index)
        train_x = train
        train_y = y

        test_x=train.iloc[test_index]
        test_y=y.iloc[test_index]
        break
    # 创建dataset
    train_loader = create_datasets(batch_size,True, train_x, train_y)
    test_loader  = create_datasets(batch_size,False,test_x,  test_y)
    # initial nn
    net=Net(dim)
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in',nonlinearity='leaky_relu')
    # Binary cross entropy loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=1e-2)
    # train nn
    net = train_model(net,train_loader, test_loader,2000,160)
    # prediction
    torch.cuda.empty_cache()
    batch_size=512
    test_sub_loader= create_datasets_prediction(batch_size,False,test)
    if model_seed==0:
        prediction = make_prediction(net,test_sub_loader)
    else :
        prediction += make_prediction(net,test_sub_loader)

# 平均到10个seed
prediction = prediction/num_model_seed

# submit = pd.read_csv("./data/提交结果示例.csv.csv")
# submit['event_id']=test_event_id
# prob_cols=['prob_{}'.format(i) for i in range(4)]
# print(prob_cols)
# prediction=pd.DataFrame(prediction,columns=prob_cols)
# prediction['id']=submit['id']
# submit=submit.merge(prediction,how='left',on='id')
# for iii in prob_cols:
#     submit[iii]= submit.groupby('event_id')[iii].transform('mean')
# submit['label']=np.argmax(submit[prob_cols].values,axis=1)
#
# dit = {0:21,1:1,2:4,3:5}
# submit['label'] = submit['label'].map(dit)
# submit.to_csv('nn_160p.csv', index=None)
# submit[['id','label']].to_csv('nn_160.csv', index=None)