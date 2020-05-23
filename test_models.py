import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time
import yaml

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

def prepare_X_y(dataset_name, dir_data, balanced_dataset=False):
    if dataset_name == 'cancer':
        from sklearn import datasets
        cancer = datasets.load_breast_cancer()
        X = cancer.data
        y = cancer.target
    elif dataset_name == 'credit':
        X, y = prepare_data(dir_data, 'creditcard.csv')
        if balanced_dataset:
            idx_is_fraud  = [i for i in range(len(y)) if y[i] == 1]
            idx_not_fraud = [i for i in range(len(y)) if y[i] == 0]
            X_is_fraud = X[idx_is_fraud]
            y_is_fraud = y[idx_is_fraud]
            X_not_fraud = X[idx_not_fraud]
            y_not_fraud = y[idx_not_fraud]
            np.random.seed(0)
            idx_rand = np.random.randint(len(X_not_fraud), size=len(X_is_fraud))
            X_not_fraud = X_not_fraud[idx_rand]
            y_not_fraud = y_not_fraud[idx_rand]
            X = np.concatenate((X_is_fraud, X_not_fraud), axis=0)
            y = np.concatenate((y_is_fraud, y_not_fraud), axis=0)
    else:
        return None, None
    return X, y

def vary_parameters(X, y, model, dict_params):
    keys = list()
    for key in dict_params.keys():
        keys.append(key)
    params1 = dict_params[keys[0]]
    params2 = dict_params[keys[1]]

    acc, tn, fp, fn, tp, f1, auc = \
        [np.zeros(len(params1) * len(params2)) for _ in range(7)]

    nb_splits = 5
    kwargs = dict()
    kwargs['random_state'] = 0
    i = 0
    for param1 in params1:
        for param2 in params2:
            cv = ShuffleSplit(n_splits=nb_splits, random_state=0)
            clf_best = None
            score_best = 0
            for train_indexes, test_indexes in cv.split(X,y):
                kwargs[keys[0]] = param1
                kwargs[keys[1]] = param2
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
            acc[i] = accuracy_score(y_true, y_pred)
            tn[i], fp[i], fn[i], tp[i] = confusion_matrix(y_true, y_pred).ravel()
            f1[i] = f1_score(y_true, y_pred)
            auc[i] = roc_auc_score(y_true, y_score)
            i+=1

    d = {
        'accuracy':acc,
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
    df.insert(0, keys[0], np.repeat(params1, len(params2)), allow_duplicates=True)
    df.insert(1, keys[1], (len(params1) * list(params2)), allow_duplicates=True)
    return df

def vary_parameters_model(X, y, kwargs):
    print('Testing parameters for model :', kwargs['model_str'])
    df = vary_parameters(X, y, kwargs['model'], kwargs['params'])
    df.insert(0, 'model', kwargs['model_str'], allow_duplicates=True)
    timestr = time.strftime('%d_%m_%Y_%H_%M')
    df.to_csv('_'.join(['./csv_dir/' + kwargs['dataset_name'], 
        kwargs['model_str'], timestr + '.csv']), index=False)

def vary_parameters_random_forest(X, y, dataset_name, dict_params):
    kwargs = dict()
    kwargs['dataset_name'] = dataset_name
    kwargs['model_str'] = 'Rand_Forest'
    kwargs['model'] = RandomForestClassifier
    kwargs['params'] = dict_params 
    vary_parameters_model(X, y, kwargs)

def vary_parameters_adaboost(X, y, dataset_name, dict_params):
    kwargs = dict()
    kwargs['dataset_name'] = dataset_name
    kwargs['model_str'] = 'AdaBoost'
    kwargs['model'] = AdaBoostClassifier
    kwargs['params'] = dict_params 
    vary_parameters_model(X, y, kwargs)

def vary_parameters_gradient_boost(X, y, dataset_name, dict_params):
    kwargs = dict()
    kwargs['dataset_name'] = dataset_name
    kwargs['model_str'] = 'GradientBoost'
    kwargs['model'] = GradientBoostingClassifier
    kwargs['params'] = dict_params 
    vary_parameters_model(X, y, kwargs)

def timeit(f):
    def timed(*args, **kwargs):
        t0 = time.time()
        ret = f(*args, **kwargs)
        print('execution time of {} : {:.2f} s'.format(f.__name__, time.time() - t0))
        return ret
    return timed

def yaml_loader(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(stream=f, Loader=yaml.FullLoader)
    return data

def run_tests(dataset_name, dir_data):
    X, y = prepare_X_y(dataset_name, dir_data, balanced_dataset=True)
    if X is None:
        print('error while preparing dataset')
        return

    dict_params = dict()
    dict_params['n_estimators'] = np.arange(50,151,50)
    dict_params['max_depth'] = np.arange(6,17,2)
    timeit(vary_parameters_random_forest)(X, y, dataset_name, dict_params)

    dict_params = dict()
    dict_params['n_estimators'] = np.arange(60,161,20)
    dict_params['learning_rate'] = np.arange(0.5,1.6,0.5)
    timeit(vary_parameters_adaboost)(X, y, dataset_name, dict_params)

    dict_params = dict()
    dict_params['n_estimators'] = np.arange(60,161,20)
    dict_params['learning_rate'] = np.arange(0.5,1.6,0.5)
    timeit(vary_parameters_gradient_boost)(X, y, dataset_name, dict_params)

if __name__ == '__main__':
    # dataset_name = 'cancer'
    # dataset_name = 'credit'

    kwargs = yaml_loader('./config.yaml')
    run_tests(**kwargs)