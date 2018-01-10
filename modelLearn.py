#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com

import xgboost as xgb
import scipy as sp
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def xgb_model(train_data, validation_data, test_data):
    y_train = train_data['label']
    x_train = train_data.drop(['label'], axis=1)
    y_val = validation_data['label']
    x_val = validation_data.drop(['label'], axis=1)
    x_test = test_data.drop(['label'], axis=1)
    num_round = 100
    dtrain = xgb.DMatrix(x_train, label=y_train, missing=0)
    dval = xgb.DMatrix(x_val, label=y_val, missing=0)
    dtest = xgb.DMatrix(x_test, missing=0)
    params = {
             'max_depth': 6,
              'eta': 0.15,
              'silent': 0,
              'scale_pos_weight': 1,
              'objective': 'binary:logistic',
              'gamma': 0,
              "min_child_weight": 1,
              "max_delta_step": 0
              }
    # param['nthread'] = 1
    params['eval_metric'] = ['logloss']
    params['subsample'] = 0.85
    params['colsample_bytree'] = 0.9
    params['base_score'] = 0.5
    params['seed'] = 0
    eval_list = [(dval, 'eval'), (dtrain, 'train')]

    # training
    xgbmodel = xgb.train(params, dtrain, num_round, eval_list, early_stopping_rounds=100)
    # save xgb model
    xgbmodel.save_model('xgb_num_round_100.model')
    # load xgb model
    # bst = xgb.Booster(model_file='xgb.model')
    predict_prob_Y = xgbmodel.predict(dval)
    test_prob_Y = xgbmodel.predict(dtest)
    clf_threshold = 0.5
    test_Y = map(lambda x: 1 if x > clf_threshold else 0, test_prob_Y)

    score = logloss(y_val, predict_prob_Y)
    print "=============score==============="
    print score
    return xgbmodel, test_prob_Y, test_Y


def xgb_lr_model(train_data, validation_data, test_data):
    y_train = train_data['label']
    x_train = train_data.drop(['label'], axis=1)
    y_val = validation_data['label']
    x_val = validation_data.drop(['label'], axis=1)
    x_test = test_data.drop(['label'], axis=1)
    x_train, train_X_lr, y_train, train_Y_lr = train_test_split(x_train, y_train, test_size=0.5)

    num_round = 800
    dtrain = xgb.DMatrix(x_train, label=y_train, missing=0)
    dtrain_lr = xgb.DMatrix(train_X_lr, label=train_Y_lr, missing=0)
    dval = xgb.DMatrix(x_val, label=y_val, missing=0)
    dtest = xgb.DMatrix(x_test, missing=0)
    params = {'max_depth': 6,
              'eta': 0.15,
              'silent': 0,
              'scale_pos_weight': 1,
              'objective': 'binary:logistic',
              'gamma': 0,
              "min_child_weight": 1,
              "max_delta_step": 0}
    # param['nthread'] = 1
    params['eval_metric'] = ['logloss']
    params['subsample'] = 0.85
    params['colsample_bytree'] = 0.9
    params['base_score'] = 0.5
    params['seed'] = 0
    eval_list = [(dval, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, eval_list, early_stopping_rounds=100)
    # save xgb model
    bst.save_model('xgb_800_random.model')
    # load xgb model
    # bst = xgb.Booster(model_file='xgb.model')

    # add LR
    bst_enc = OneHotEncoder()
    bst_lm = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.01, fit_intercept=True, class_weight=None,
                                intercept_scaling=1, solver='liblinear', max_iter=100)
    leafTrainNodeInfo = bst.predict(dtrain_lr, pred_leaf=True)
    leafValNodeInfo = bst.predict(dval, pred_leaf=True)
    leafTestNodeInfo = bst.predict(dtest, pred_leaf=True)
    bst_enc.fit(leafTrainNodeInfo)
    _train_X_lr = bst_enc.transform(leafTrainNodeInfo)
    _val_X_lr = bst_enc.transform(leafValNodeInfo)
    _test_X = bst_enc.transform(leafTestNodeInfo)

    bst_lm.fit(_train_X_lr, train_Y_lr)
    model = bst_lm
    predict_prob_Y = model.predict_proba(_val_X_lr)[:, 1]

    test_prob_Y = model.predict_proba(_test_X)[:, 1]
    test_Y = model.predict(_test_X)

    score = logloss(y_val, predict_prob_Y)
    print "=============score==============="
    print score
    return bst, model, test_prob_Y, test_Y


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
