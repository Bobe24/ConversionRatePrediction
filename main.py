#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com

from dataLoad import data_preprocess
from modelLearn import xgb_model, xgb_lr_model
import pandas as pd
import zipfile
import xgboost as xgb
import sys


def get_submission(model_name):
    train_data, validation_data, test_data = data_preprocess(ohe=True, std=False)
    if model_name == 'xgb':
        model_xgb, predict_prob_Y, predict_Y = xgb_model(train_data, validation_data, test_data)
    elif model_name == 'xgb_lr':
        model_xgb, model_lr, predict_prob_Y, predict_Y = xgb_lr_model(train_data, validation_data, test_data)

    # get submission
    resultDataFrame = pd.DataFrame()
    resultDataFrame['instanceID'] = pd.Series(range(1, len(predict_prob_Y) + 1))
    resultDataFrame['prob'] = pd.Series(predict_prob_Y)
    resultDataFrame.sort_values("instanceID", inplace=True)
    savePath = "submission.csv"
    resultDataFrame.to_csv(savePath, index=False)
    with zipfile.ZipFile("submission.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
    # get feature importance
    feature_importances = get_xgb_feat_importances(model_xgb)
    savePath = "feature_importances.csv"
    feature_importances.to_csv(savePath, index=False)


def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()

    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    return feat_importances


if __name__ == "__main__":
    get_submission(model_name=sys.argv[1])
