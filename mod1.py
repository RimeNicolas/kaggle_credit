import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

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

if __name__ == '__main__':
    df = load_data()
    print(df.head())