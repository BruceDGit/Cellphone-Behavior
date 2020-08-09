import pandas  as pd
import numpy as np

def ensemble_all():
    proba1 = pd.read_csv('result/submit_proba1.csv')
    proba2 = pd.read_csv('result/submit_proba2.csv')
    proba3 = pd.read_csv('result/submit_proba3.csv').iloc[:, 1:]
    proba4 = pd.read_csv('result/submit_proba4.csv').iloc[:, 1:]

    proba1 = np.array(proba1)
    proba2 = np.array(proba2)
    proba3 = np.array(proba3)
    proba4 = np.array(proba4)

    proba = proba1 * 0.349 + proba2 * 0.120 + proba3 * 0.355 + proba4 * 0.176

    sub = pd.read_csv('data/提交结果示例.csv')
    sub.behavior_id = np.argmax(proba, axis=1)
    sub.to_csv('result/submit.csv', index=False)
