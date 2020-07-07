import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import gc
import os
from tqdm import tqdm
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from itertools import combinations


data_path = '../data/'
data_train = pd.read_csv(data_path+'sensor_train.csv')
data_test = pd.read_csv(data_path+'sensor_test.csv')
data_test['fragment_id'] += 10000
label = 'behavior_id'

data = pd.concat([data_train, data_test], sort=False)
base_fea=['time_point','acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg']
for col in base_fea:
    data['{}_rank'.format(col)] = data[col].rank()


print("加减乘除...")
for a, b in tqdm(combinations(base_fea, 2)):
    data[str(a) + '+' + str(b)] = data[a].astype(float) + data[b].astype(float)
    data[str(a) + '-' + str(b)] = data[a].astype(float) - data[b].astype(float)
    data[str(a) + '*' + str(b)] = data[a].astype(float) * data[b].astype(float)
    data[str(a) + '/' + str(b)] = data[a].astype(float) / (data[b].astype(float) + 1)
    # df[str(a) + '/log' + str(b)] = df[a].astype(float) / np.log1p(df[b].astype(float))
# # 类别特征计数
# print("类别特征计数...")
# for i in tqdm(base_fea):
#     data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['fragment_id'].transform('count')

# # 类别特征nunqiue
# print("类别特征nunqiue...")
# for i in tqdm(base_fea):
#     for j in base_fea:
#         if i != j:
#             base_fea['nuni_{0}_{1}'.format(i, j)] = base_fea[i].map(base_fea.groupby(i)[j].nunique())

def get_cvr_fea(df):
    cat_list = ['time_point','acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg']
    print("cat_list", cat_list)

    # 类别特征五折转化率特征
    print("转化率特征....")
    df['ID'] = df.index
    df['fold'] = df['ID'] % 5
    df.loc[df.behavior_id.isnull(), 'fold'] = 5
    target_feat = []
    for i in tqdm(cat_list):
        target_feat.extend([i + '_mean_last_1'])
        df[i + '_mean_last_1'] = None
        for fold in range(6):
            df.loc[df['fold'] == fold, i + '_mean_last_1'] = df[df['fold'] == fold][i].map(
                df[(df['fold'] != fold) & (df['fold'] != 5)].groupby(i)['behavior_id'].mean()
            )
        df[i + '_mean_last_1'] = df[i + '_mean_last_1'].astype(float)

    return df
data = get_cvr_fea(data)

df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]
data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
for f in tqdm(base_fea):
    for stat in ['min', 'max', 'mean', 'median', 'std', 'skew','var','sum']:
        df[f+'_'+stat] = data.groupby('fragment_id')[f].agg(stat).values

train_df = df[df[label].isna()==False].reset_index(drop=True)
test_df = df[df[label].isna()==True].reset_index(drop=True)

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
print(len(used_feat))
print(used_feat)


train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]


scores = []
imp = pd.DataFrame()
imp['feat'] = used_feat

params = {
    'learning_rate': 0.1,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'n_jobs': 4,
    'seed': 2020,
    'max_depth': 10,
    'num_leaves': 64,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
}



oof_train = np.zeros((len(train_x), 19))
preds = np.zeros((len(test_x), 19))
folds = 5
seeds = [44]#, 2020, 527, 1527]
for seed in seeds:
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[val_idx]
        train_set = lgb.Dataset(x_trn, y_trn)
        val_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params, train_set, num_boost_round=500000,
                          valid_sets=(train_set, val_set), early_stopping_rounds=50,
                          verbose_eval=20)
        oof_train[val_idx] += model.predict(x_val) / len(seeds)
        preds += model.predict(test_x) / folds / len(seeds)
        scores.append(model.best_score['valid_1']['multi_error'])
        imp['gain' + str(fold + 1)] = model.feature_importance(importance_type='gain')
        imp['split' + str(fold + 1)] = model.feature_importance(importance_type='split')
        del x_trn, y_trn, x_val, y_val, model, train_set, val_set
        gc.collect()


imp['gain'] = imp[[f for f in imp.columns if 'gain' in f]].sum(axis=1)/folds
imp['split'] = imp[[f for f in imp.columns if 'split' in f]].sum(axis=1)
imp = imp.sort_values(by=['gain'], ascending=False)
imp[['feat', 'gain', 'split']]
imp = imp.sort_values(by=['split'], ascending=False)
imp[['feat', 'gain', 'split']]



def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

labels = np.argmax(preds, axis=1)
oof_y = np.argmax(oof_train, axis=1)


round(accuracy_score(train_y, oof_y), 5)

score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(train_y, oof_y)) / oof_y.shape[0]
round(score, 5)



sub = pd.read_csv(data_path+'提交结果示例.csv')

sub['behavior_id'] = labels

vc = data_train['behavior_id'].value_counts().sort_index()
sns.barplot(vc.index, vc.values)
plt.show()
vc = sub['behavior_id'].value_counts().sort_index()
sns.barplot(vc.index, vc.values)
plt.show()
sub.to_csv('result/sub%.5f.csv' % score, index=False)