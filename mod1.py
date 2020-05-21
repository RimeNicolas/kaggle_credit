import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def load_data():
    with open('path_data.txt', 'r') as f:
        dir_data = f.read()
    return pd.read_csv(os.path.join(dir_data, 'creditcard.csv'))

def build_dict_feature(df):
    dict_feature = dict()
    for i in range(1,29):
        l = list(df['V'+str(i)])
        dict_feature[i] = l / np.linalg.norm(l)
    return dict_feature

def build_array_feature(df):
    array_feature = list()
    for i in range(1,29):
        l = list(df['V'+str(i)])
        array_feature.append(l / np.linalg.norm(l))
    return np.array(array_feature)

def prepare_data():
    df = load_data()
    X = build_array_feature(df).T
    y = np.array(df['Class'])
    return X, y

def plot_roc(parameter, list_fpr_rt, list_tpr_rt):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(len(parameter)):
        plt.plot(list_fpr_rt[i], list_tpr_rt[i], label=parameter[i])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def run_test(X, y, model, keys, params1, params2):
    scores = np.zeros(len(params1) * len(params2))
    tn = np.zeros(len(params1) * len(params2))
    fp = np.zeros(len(params1) * len(params2))
    fn = np.zeros(len(params1) * len(params2))
    tp = np.zeros(len(params1) * len(params2))
    f1 = np.zeros(len(params1) * len(params2))
    auc = np.zeros(len(params1) * len(params2))

    i = 0
    for param1 in params1:
        for param2 in params2:
            nb_splits = 5
            cv = ShuffleSplit(n_splits=nb_splits, random_state=0)
            #skf = StratifiedKFold(n_splits=nb_splits)
            clf_best = None
            score_best = 0
            for train_indexes, test_indexes in cv.split(X,y):
                kwargs = dict()
                kwargs[keys[0]] = param1
                kwargs[keys[1]] = param2
                kwargs['random_state'] = 0
                clf = model(**kwargs)
                clf.fit(X[train_indexes], y[train_indexes])
                score = clf.score(X[test_indexes], y[test_indexes])
                if score > score_best:
                    score_best = score
                    clf_best = clf
            clf = None
            y_pred = clf_best.predict(X[test_indexes])
            y_score = clf_best.predict_proba(X[test_indexes])[:,1]
            y_true = y[test_indexes]
            scores[i] = score_best
            tn[i], fp[i], fn[i], tp[i] = confusion_matrix(y_true, y_pred).ravel()
            f1[i] = f1_score(y_true, y_pred)
            auc[i] = roc_auc_score(y_true, y_score)
            i+=1

    d = {
        'accuracy':scores,
        'TP':tp,
        'FP':fp,
        'TN':tn,
        'FN':fn,
        'True_Positive_rate':tp/(tp + fn),
        'False_Positive_rate':np.ones(len(tn)) - tn/(tn + fp),
        'precision':tp/(tp + fp),
        'F1_score':f1,
        'AUC':auc
    }
    df = pd.DataFrame(data=d)
    return df

if __name__ == '__main__':
    nb_trees = np.arange(50,151,50)
    max_depths = np.arange(1,5,1)
    X, y = prepare_data()
    # from sklearn import datasets
    # cancer = datasets.load_breast_cancer()
    # X = cancer.data
    # y = cancer.target
    model = RandomForestClassifier
    keys = ['n_estimators', 'max_depth']
    df = run_test(X, y, model, keys, nb_trees, max_depths)
    df.insert(0, keys[0], np.repeat(nb_trees, len(max_depths)), allow_duplicates=True)
    df.insert(1, keys[1], (len(nb_trees) * list(max_depths)), allow_duplicates=True)
    df.to_csv('credit_scores.csv', index=False)