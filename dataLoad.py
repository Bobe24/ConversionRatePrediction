#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com

import pandas as pd
import numpy as np
from config import train_path, test_path, user_path, position_path, user_app_action_path, user_installedapps_path, \
    ad_path, app_categories_path, enc_clm, numer_col_names, train_duplicate_path, test_duplicate_path, \
    app_actions_count_path, app_installed_count_path, user_actions_count_path, user_installed_count_path,\
    train_cor_path, test_cor_path, train_td_path, test_td_path,position_connectionType,advertiser_position,\
    gender_position,hometown_residence,age_marriageStatus,age_position,app_position,hometown_position,\
    position_telecomsOperator, creativeID_positionID ,education_positionID_path,campaignID_positionID ,\
    marriageStatus_positionID,residence_positionID ,age_telecomsOperator ,age_education ,camgaignID_connectionType,\
    gender_education,marriageStatus_residence
from dataGen import user_latest_install
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, Imputer


def data_preprocess(ohe=False, std=False):
    train_data = get_data(train_path)
    test_data = get_data(test_path)
    ad_data = get_data(ad_path)
    user_data = get_data(user_path)
    position_data = get_data(position_path)
    app_categories_data = get_data(app_categories_path)
    user_app_action = get_data(user_app_action_path)
    # user_installedapps_data = get_data(user_installedapps_path)
    train_duplicate_data = get_data(train_duplicate_path)
    test_duplicate_data = get_data(test_duplicate_path)
    app_actions_count_data = get_data(app_actions_count_path)
    app_installed_count_data = get_data(app_installed_count_path)
    user_actions_count_data = get_data(user_actions_count_path)
    user_installed_count_data = get_data(user_installed_count_path)
    train_td = get_data(train_td_path)
    test_td = get_data(test_td_path)

    # 添加重复点击特征
    train_data = pd.concat([train_data, train_duplicate_data], axis=1)
    test_data = pd.concat([test_data, test_duplicate_data], axis=1)
    # 添加时序特征
    train_data = pd.concat([train_data, train_td], axis=1)
    test_data = pd.concat([test_data, test_td], axis=1)

    # 过滤数据
    train_data = train_data.drop(['conversionTime'], axis=1)
    test_data = test_data.drop(['instanceID'], axis=1)
    # 合并训练数据和测试数据
    concat_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 使用user_app_actions文件得到每个user之前最近安装的app所属的app category
    concat_data = user_latest_install(concat_data, user_app_action, app_categories_data)
    # 使用user_app_actions文件统计每个app过去16天的安装量
    concat_data = pd.concat([concat_data, app_actions_count_data], axis=1)
    # 使用user_installedapps文件统计每个app的安装量
    concat_data = pd.concat([concat_data, app_installed_count_data], axis=1)
    # 使用user_app_actions文件统计每个user过去16天的安装量
    concat_data = pd.concat([concat_data, user_actions_count_data], axis=1)
    # 使用user_installedapps文件统计每个user的安装量
    concat_data = pd.concat([concat_data, user_installed_count_data], axis=1)

    # 时间转化为日期
    concat_data['clickDate'] = map(int, (concat_data['clickTime'] / 10000))
    # 时间转化为星期
    concat_data['clickWeek'] = map(int, (concat_data['clickTime'] / 10000 % 7))
    # 是否周末
    concat_data['isWeekend'] = map(lambda x: 1 if (x == 4 or x == 5) else 0, concat_data['clickWeek'])
    # 时间转化为小时
    concat_data['clickHour'] = map(int, (concat_data['clickTime'] / 100 % 100))
    # 小时转化为是否闲暇时间
    concat_data['highConversionHour'] = map(lambda x: 1 if (9 <= x <= 12) else 0, concat_data['clickHour'])

    # 合并user数据
    concat_data = pd.merge(concat_data, user_data, how='left', on='userID')
    # 用户年龄分段
    concat_data['ageClass'] = map(int, (concat_data['age'] / 5))
    # 年龄性别特征组合
    concat_data['age_gender'] = concat_data['ageClass'] + concat_data['gender'] * 100

    # 组合数据
    concat_data = pd.merge(concat_data, ad_data, how='left', on='creativeID')
    concat_data = pd.merge(concat_data, position_data, how='left', on='positionID')
    concat_data = pd.merge(concat_data, app_categories_data, how='left', on='appID')

    # 判断当前appID与用户最近安装的latest_install_appID是否同一类别
    concat_data['same_appCate'] = map(lambda x, y: 1 if (x == y) else 0, concat_data['appCategory'],  concat_data['latest_install_appCategory'])
    concat_data = concat_data.drop(['latest_install_appCategory'], axis=1)

    # reset DataFrame index
    concat_data_min = concat_data[concat_data['clickDate'] >= 22].reset_index()

    # 交叉特征
    position_connectionType_data = get_data(position_connectionType)
    advertiser_position_data = get_data(advertiser_position)
    gender_position_data = get_data(gender_position)
    hometown_residence_data = get_data(hometown_residence)
    age_marriageStatus_data = get_data(age_marriageStatus)
    age_position_data = get_data(age_position)
    app_position_data = get_data(app_position)
    hometown_position_data = get_data(hometown_position)
    position_telecomsOperator_data = get_data(position_telecomsOperator)
    creativeID_positionID_data = get_data(creativeID_positionID)
    education_positionID_data = get_data(education_positionID_path)
    campaignID_positionID_data = get_data(campaignID_positionID)
    marriageStatus_positionID_data = get_data(marriageStatus_positionID)
    residence_positionID_data = get_data(residence_positionID)
    age_telecomsOperator_data = get_data(age_telecomsOperator)
    age_education_data = get_data(age_education)
    camgaignID_connectionType_data = get_data(camgaignID_connectionType)
    gender_education_data = get_data(gender_education)
    marriageStatus_residence_data = get_data(marriageStatus_residence)

    concat_data_min = pd.concat([concat_data_min, position_connectionType_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, advertiser_position_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, gender_position_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, hometown_residence_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, age_marriageStatus_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, age_position_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, app_position_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, hometown_position_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, position_telecomsOperator_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, creativeID_positionID_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, education_positionID_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, campaignID_positionID_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, marriageStatus_positionID_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, residence_positionID_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, age_telecomsOperator_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, age_education_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, camgaignID_connectionType_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, gender_education_data], axis=1)
    concat_data_min = pd.concat([concat_data_min, marriageStatus_residence_data], axis=1)
    # drop 一些属性
    concat_data_min = concat_data_min.drop(['clickTime', 'age'], axis=1)
    # drop userID and positionID to test
    # concat_data_min = concat_data_min.drop(['userID', 'positionID'], axis=1)
    concat_data_min.fillna(0, inplace=True)
    # 变量one-hot编码
    if ohe:
        concat_data_min = cat_features_one_hot_encode(concat_data_min)
    # 数值变量标准化
    if std:
        concat_data_min = num_features_standar_scaler(concat_data_min)
    # 拆分训练集和测试集
    train_data, validation_data, test_data = data_split(concat_data_min)
    return train_data, validation_data, test_data


def get_data(path):
    data = pd.read_csv(path, sep=',', na_values=[None, np.nan])
    return data


def cat_features_one_hot_encode(row_data_frame):
    return pd.get_dummies(row_data_frame, dummy_na=True, columns=enc_clm)


def num_features_standar_scaler(row_data_frame, std_method="std"):
    """
    0-1标准化 和 z-score标准化数据
    :param row_data_frame:
    :return:
    """
    imp = Imputer(missing_values=np.nan, strategy="mean", axis=0)
    imp.fit(row_data_frame[numer_col_names].values)
    row_data_frame[numer_col_names] = imp.transform(row_data_frame[numer_col_names].values)  # 对训练集数值型特征进行均值填充
    if std_method == "minmax":
        min_max = MinMaxScaler()
        min_max_scaler = min_max.fit(row_data_frame[numer_col_names].values)
        row_data_frame[numer_col_names] = min_max_scaler.transform(row_data_frame[numer_col_names].values)
        minxmax_scale_train_df = row_data_frame
        return minxmax_scale_train_df
    elif std_method == "std":
        std = StandardScaler()
        std_scaler = std.fit(row_data_frame[numer_col_names].values)
        row_data_frame[numer_col_names] = std_scaler.transform(row_data_frame[numer_col_names].values)
        std_scale_train_df = row_data_frame
        return std_scale_train_df


def data_split(concat_data):
    train_data = concat_data[(concat_data['clickDate'] >= 22) & (concat_data['clickDate'] <= 29)]
    validation_data = concat_data[concat_data['clickDate'] == 29]
    test_data = concat_data[concat_data['label'] == -1]
    return train_data, validation_data, test_data
