# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:18:09 2023

@author: zhua079
"""
import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


signal_size = 1024
work_condition = ['_0.mat', '_1.mat', '_2.mat', '_3.mat']
dataname = {0: [os.path.join('normal'+work_condition[0]),
                os.path.join('IR007'+work_condition[0]), os.path.join(
    'OR007@6'+work_condition[0]), os.path.join('B007'+work_condition[0]),
    os.path.join('IR014'+work_condition[0]), os.path.join(
    'OR014@6'+work_condition[0]), os.path.join('B014'+work_condition[0]),
    os.path.join('IR021'+work_condition[0]), os.path.join('OR021@6'+work_condition[0]), os.path.join('B021'+work_condition[0])],  # 1797rpm
    1: [os.path.join('normal'+work_condition[1]),
        os.path.join('IR007'+work_condition[1]), os.path.join(
        'OR007@6'+work_condition[1]), os.path.join('B007'+work_condition[1]),
        os.path.join('IR014'+work_condition[1]), os.path.join(
        'OR014@6'+work_condition[1]), os.path.join('B014'+work_condition[1]),
        os.path.join('IR021'+work_condition[1]), os.path.join('OR021@6'+work_condition[1]), os.path.join('B021'+work_condition[1])],  # 1772rpm
    2: [os.path.join('normal'+work_condition[2]),
        os.path.join('IR007'+work_condition[2]), os.path.join(
        'OR007@6'+work_condition[2]), os.path.join('B007'+work_condition[2]),
        os.path.join('IR014'+work_condition[2]), os.path.join(
        'OR014@6'+work_condition[2]), os.path.join('B014'+work_condition[2]),
        os.path.join('IR021'+work_condition[2]), os.path.join('OR021@6'+work_condition[2]), os.path.join('B021'+work_condition[2])],  # 1750rpm
    3: [os.path.join('normal'+work_condition[3]),
        os.path.join('IR007'+work_condition[3]), os.path.join(
        'OR007@6'+work_condition[3]), os.path.join('B007'+work_condition[3]),
        os.path.join('IR014'+work_condition[3]), os.path.join(
        'OR014@6'+work_condition[3]), os.path.join('B014'+work_condition[3]),
        os.path.join('IR021'+work_condition[3]), os.path.join('OR021@6'+work_condition[3]), os.path.join('B021'+work_condition[3])]}  # 1730rpm

axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]


def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, dataname[N[k]][n])
            else:
                path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def get_files1(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, dataname[N[k]][n])
            else:
                path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    # datanumber = axisname.split(".")
    realaxis = np.array(list(loadmat(filename).items()), dtype=object)[
        3, -2].split("_")[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    overlap = 256
    data = []
    lab = []
    start, end = 0, signal_size
    count = 0
    while count < 100:  # fl.shape[0]- signal_size ,  signal_size*100
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size - overlap
        end += signal_size - overlap
        count += 1
    return data, lab
