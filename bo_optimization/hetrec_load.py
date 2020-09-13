#!/usr/bin/env python
# coding: utf-8
"""Hetrec 2011 Data loading"""

# !pip install numpy_indexed
# !wget -c http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
# !unzip hetrec2011-lastfm-2k.zip
# !rm hetrec2011-lastfm-2k.zip
# !ls


import pandas as pd
import numpy as np


def apply_fromiter1D(arr, f):
    return np.fromiter((f(xi) for xi in arr), arr.dtype)


def make_full(data):
    unique_rows = np.unique(data[:, 0])
    unique_cols = np.unique(data[:, 1])
    dict_fun = lambda d: (lambda x: d[x])

    d_rows = dict(zip(unique_rows, range(unique_rows.size)))
    d_cols = dict(zip(unique_cols, range(unique_cols.size)))
    data_full = np.zeros((unique_rows.size, unique_cols.size))
    data_full[np.array([d_rows[x] for x in data[:, 0]]), np.array([d_cols[x] for x in data[:, 1]])] = data[:, -1]

    data_new = data.copy()
    data_new[:, 0] = apply_fromiter1D(data[:, 0], dict_fun(d_rows))
    data_new[:, 1] = apply_fromiter1D(data[:, 1], dict_fun(d_cols))
    return data_full, data_new


def load_train_test(filen='../data/user_artists.dat', frac=0.9):
    data = pd.read_csv(filen, sep="\t").values
    datafull, data = make_full(data)
        
    train_path = filen+"_train%s.npy" % frac
    test_path = filen+"_test%s.npy" % frac

    try:
      print("[load_train_test] Loading train from %s and test from %s" % (train_path, test_path))
      data_train = np.load(train_path)
      data_test = np.load(test_path)

    except Exception as e:
      print("[load_train_test] FAILED:", e)
      print("[load_train_test] WARNING: Recomputing train-test split!")

      tempd = pd.DataFrame(data).groupby(by=0).apply(lambda x: x.sample(frac=frac)).reset_index(level=1)['level_1'].values
      data_train = data[tempd, :]
      data_test = data[~tempd, :]

      np.save(train_path, data_train)
      np.save(test_path, data_test)
    

    return data, datafull, data_train, data_test


def train_test_resample(data, frac=0.9, random_state=123):
    tempd = pd.DataFrame(data).groupby(by=0).apply(lambda x: x.sample(frac=frac, random_state=random_state)).reset_index(level=1)['level_1'].values
    data_train = data[tempd, :]
    data_test = data[~tempd, :]

    return data, data_train, data_test

