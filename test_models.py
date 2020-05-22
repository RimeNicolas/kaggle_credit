import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time
import yaml

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def load_data(dir_data, csv_name):
    return pd.read_csv(os.path.join(dir_data, csv_name))

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

def prepare_data(dir_data, csv_name):
    df = load_data(dir_data, csv_name)
    X = build_array_feature(df).T
    y = np.array(df['Class'])
    return X, y

def test_parameters(X, y, model, keys, params1, params2):
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

def prepare_X_y(dataset_name, dir_data):
    if dataset_name == 'cancer':
        from sklearn import datasets
        cancer = datasets.load_breast_cancer()
        X = cancer.data
        y = cancer.target
    elif dataset_name == 'credit':
        X, y = prepare_data(dir_data, 'creditcard.csv')
    else:
        return
    return X, y

def test_parameters_random_forest(dataset_name, X,y):
    model = RandomForestClassifier
    model_str = 'Rand_Forest'
    keys = ['n_estimators', 'max_depth']
    nb_trees = np.arange(50,101,50)
    max_depths = np.arange(6,13,2)
    print('Testing parameters for model :', model_str)
    df = test_parameters(X, y, model, keys, nb_trees, max_depths)
    df.insert(0, 'model', model_str, allow_duplicates=True)
    df.insert(1, keys[0], np.repeat(nb_trees, len(max_depths)), allow_duplicates=True)
    df.insert(2, keys[1], (len(nb_trees) * list(max_depths)), allow_duplicates=True)
    timestr = time.strftime('%d_%m_%Y_%H_%M')
    df.to_csv('_'.join(['./csv_dir/' + dataset_name, model_str, timestr + '.csv']), index=False)

def test_parameters_adaboost(dataset_name, X, y):
    model = AdaBoostClassifier
    model_str = 'AdaBoost'
    keys = ['n_estimators', 'learning_rate']
    n_estimators = np.arange(100,161,20)
    learning_rates = np.arange(0.5,1.6,0.5)
    print('Testing parameters for model :', model_str)
    df = test_parameters(X, y, model, keys, n_estimators, learning_rates)
    df.insert(0, 'model', model_str, allow_duplicates=True)
    df.insert(1, keys[0], np.repeat(n_estimators, len(learning_rates)), allow_duplicates=True)
    df.insert(2, keys[1], (len(n_estimators) * list(learning_rates)), allow_duplicates=True)
    timestr = time.strftime('%d_%m_%Y_%H_%M')
    df.to_csv('_'.join(['./csv_dir/' + dataset_name, model_str, timestr + '.csv']), index=False)

def timeit(f):
    def timed(*args, **kwargs):
        t0 = time.time()
        ret = f(*args, **kwargs)
        print('execution time of {} : {:.2f} s'.format(f.__name__, time.time() - t0))
        return ret
    return timed

def run_tests(dataset_name, dir_data):
    X, y = prepare_X_y(dataset_name, dir_data)
    if X is None:
        return
    timeit(test_parameters_random_forest)(dataset_name, X, y)
    timeit(test_parameters_adaboost)(dataset_name, X, y)

def yaml_loader(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(stream=f, Loader=yaml.FullLoader)
    return data

if __name__ == '__main__':
    # dataset_name = 'cancer'
    # dataset_name = 'credit'

    kwargs = yaml_loader('./config.yaml')
    run_tests(**kwargs)