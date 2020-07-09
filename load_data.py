import pandas as pd


train_sequences=list()
base_fea=[]

for index,group in train.groupby(by='fragment_id'):
    train_sequences.append(group[base_fea].values)
train_sequences[0]
y_train=train.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)['behavior_id'].values
