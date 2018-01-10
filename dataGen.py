#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com

import pandas as pd
import numpy as np


# 使用user_app_actions文件得到每个user之前最近安装的app
def user_latest_install(train_data, user_app_action, app_categories_data):
    key = 'userID'
    df_train = pd.DataFrame()
    for day in range(1, 16):
        day_start = day * 10000
        day_end = day_start + 160000
        temp_data = user_app_action[
            (user_app_action['installTime'] >= day_start) & (
                user_app_action['installTime'] < day_end)]
        df_latest_install = temp_data.drop_duplicates([key], keep='last')
        df_latest_install = df_latest_install.drop(['installTime'], axis=1)
        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = train_data[
            (train_data['clickTime'] >= day_start_train) & (
                train_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_latest_install, how='left', on=key)
        df_train = pd.concat([df_train, temp_data_train], axis=0,
                             ignore_index=True)
    df_train['appID'].fillna(0, inplace=True)
    df_train = pd.merge(df_train, app_categories_data, how='left', on='appID')
    df_train = df_train.drop(['appID'], axis=1)
    df_train.rename(columns={'appCategory': 'latest_install_appCategory'}, inplace=True)
    df_train['latest_install_appCategory'].fillna(0, inplace=True)
    return df_train


# 使用user_app_actions文件统计每个app过去16天的安装量
def app_actions_count(train_data, user_app_action):
    key = "appID"
    window = 160000
    df_train = pd.DataFrame()
    for day in range(1, 16):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = user_app_action[
            (user_app_action['installTime'] >= day_start) & (user_app_action['installTime'] < day_end)]
        df_app_count = temp_data.groupby(key).apply(lambda df: np.count_nonzero(df[key])).reset_index()
        df_app_count.columns = [key, 'app_actions_count']
        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = train_data[
            (train_data['clickTime'] >= day_start_train) & (train_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_app_count, how='left', on=key)
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)
    df_train['app_actions_count'].fillna(0, inplace=True)
    df_save = df_train['app_actions_count']
    df_save.columns = ['app_actions_count']
    print df_save
    savePath = "app_actions_count_past16days.csv"
    df_save.to_csv(savePath, index=False, header=True)


# 使用user_installedapps文件统计每个app的安装量
def app_installed_count(train_data, user_installedapps_data):
    key = 'appID'
    df_user_installedapps_count = user_installedapps_data.groupby(key).apply(
        lambda df: np.count_nonzero(df[key])).reset_index()
    df_user_installedapps_count.columns = [key, 'app_installed_count']
    train_data = pd.merge(train_data, df_user_installedapps_count, how='left', on=key)
    train_data['app_installed_count'].fillna(0, inplace=True)
    df_save = train_data['app_installed_count']
    df_save.columns = ['app_installed_count']
    savePath = "app_installed_count.csv"
    df_save.to_csv(savePath, index=False, header=True)


# 使用user_app_actions文件统计每个user过去16天的安装量
def user_actions_count(train_data, user_app_action):
    key = 'userID'
    window = 160000
    df_train = pd.DataFrame()
    for day in range(1, 16):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = user_app_action[
            (user_app_action['installTime'] >= day_start) & (
                user_app_action['installTime'] < day_end)]
        df_user_count = temp_data.groupby(key).apply(
            lambda df: np.count_nonzero(df[key])).reset_index()
        df_user_count.columns = [key, 'user_actions_count']
        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = train_data[
            (train_data['clickTime'] >= day_start_train) & (
                train_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_user_count, how='left', on=key)
        df_train = pd.concat([df_train, temp_data_train], axis=0,
                             ignore_index=True)
    df_train['user_actions_count'].fillna(0, inplace=True)
    df_save = df_train['user_actions_count']
    df_save.columns = ['user_actions_count']
    print df_save
    savePath = "user_actions_count_past16days.csv"
    df_save.to_csv(savePath, index=False, header=True)


# 使用user_installedapps文件统计每个user的安装量
def user_installed_count(train_data, user_installedapps_data):
    key = 'userID'
    df_user_installed_count = user_installedapps_data.groupby(key).apply(
        lambda df: np.count_nonzero(df[key])).reset_index()
    df_user_installed_count.columns = [key, 'user_installed_count']
    train_data = pd.merge(train_data, df_user_installed_count, how='left', on=key)
    train_data['user_installed_count'].fillna(0, inplace=True)
    df_save = train_data['user_installed_count']
    df_save.columns = ['user_installed_count']
    print df_save
    savePath = "user_installed_count.csv"
    df_save.to_csv(savePath, index=False, header=True)


# 计算position_connectionType_count
def position_connectionType_count(concat_data):
    key1 = 'positionID'
    key2 = 'connectionType'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]
        df_posi_conn_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_posi_conn_count.columns = [key1, key2, 'posi_conn_count']
        df_posi_conn_conversion_rate = temp_data.groupby([key1, key2]).apply(lambda df: np.mean(df['label'])).reset_index()
        df_posi_conn_conversion_rate.columns = [key1, key2, 'posi_conn_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_posi_conn_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_posi_conn_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['posi_conn_count'].fillna(0, inplace=True)
    df_train['posi_conn_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['posi_conn_count', 'posi_conn_conversion_rate']]
    df_save.columns = ['posi_conn_count', 'posi_conn_conversion_rate']
    savePath = "position_connectionType.csv"
    df_save.to_csv(savePath, index=False, header=True)


# advertiserID_position_count
def advertiser_position_count(concat_data):
    key1 = 'positionID'
    key2 = 'advertiserID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]
        df_posi_advertiser_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_posi_advertiser_count.columns = [key1, key2, 'position_advertiser_count']

        df_posi_advertiser_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_posi_advertiser_conversion_rate.columns = [key1, key2, 'position_advertiser_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_posi_advertiser_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_posi_advertiser_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['position_advertiser_count'].fillna(0, inplace=True)
    df_train['position_advertiser_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['position_advertiser_count', 'position_advertiser_conversion_rate']]
    df_save.columns = ['position_advertiser_count', 'position_advertiser_conversion_rate']
    savePath = "advertiser_position.csv"
    df_save.to_csv(savePath, index=False, header=True)


# gender_positionID
def gender_position_count(concat_data):
    key1 = 'gender'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]
        df_gender_position_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_gender_position_count.columns = [key1, key2, 'gender_position_count']

        df_gender_position_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_gender_position_conversion_rate.columns = [key1, key2, 'gender_position_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_gender_position_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_gender_position_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['gender_position_count'].fillna(0, inplace=True)
    df_train['gender_position_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['gender_position_count', 'gender_position_conversion_rate']]
    df_save.columns = ['gender_position_count', 'gender_position_conversion_rate']
    savePath = "gender_position.csv"
    df_save.to_csv(savePath, index=False, header=True)


# hometown_residence
def hometown_residence_count(concat_data):
    key1 = 'hometown'
    key2 = 'residence'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_hometown_residence_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_hometown_residence_count.columns = [key1, key2, 'hometown_residence_count']

        df_hometown_residence_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_hometown_residence_conversion_rate.columns = [key1, key2, 'hometown_residence_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_hometown_residence_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_hometown_residence_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['hometown_residence_count'].fillna(0, inplace=True)
    df_train['hometown_residence_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['hometown_residence_count', 'hometown_residence_conversion_rate']]
    df_save.columns = ['hometown_residence_count', 'hometown_residence_conversion_rate']
    savePath = "hometown_residence.csv"
    df_save.to_csv(savePath, index=False, header=True)


# age_marriageStatus
def age_marriage_count(concat_data):
    key1 = 'age'
    key2 = 'marriageStatus'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_age_marriage_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_age_marriage_count.columns = [key1, key2, 'age_marriage_count']

        df_age_marriage_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_age_marriage_conversion_rate.columns = [key1, key2, 'age_marriage_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_age_marriage_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_age_marriage_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['age_marriage_count'].fillna(0, inplace=True)
    df_train['age_marriage_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['age_marriage_count', 'age_marriage_conversion_rate']]
    df_save.columns = ['age_marriage_count', 'age_marriage_conversion_rate']
    savePath = "age_marriageStatus.csv"
    df_save.to_csv(savePath, index=False, header=True)


# age_positionID
def age_position_count(concat_data):
    key1 = 'age'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_age_position_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_age_position_count.columns = [key1, key2, 'age_position_count']

        df_age_position_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_age_position_conversion_rate.columns = [key1, key2, 'age_position_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_age_position_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_age_position_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['age_position_count'].fillna(0, inplace=True)
    df_train['age_position_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['age_position_count', 'age_position_conversion_rate']]
    df_save.columns = ['age_position_count', 'age_position_conversion_rate']
    savePath = "age_position.csv"
    df_save.to_csv(savePath, index=False, header=True)


# appID_positionID
def app_position_count(concat_data):
    key1 = 'appID'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_app_position_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_app_position_count.columns = [key1, key2, 'app_position_count']

        df_app_position_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_app_position_conversion_rate.columns = [key1, key2, 'app_position_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_app_position_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_app_position_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['app_position_count'].fillna(0, inplace=True)
    df_train['app_position_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['app_position_count', 'app_position_conversion_rate']]
    df_save.columns = ['app_position_count', 'app_position_conversion_rate']
    savePath = "app_position.csv"
    df_save.to_csv(savePath, index=False, header=True)


# hometown_positionID
def hometown_position_count(concat_data):
    key1 = 'hometown'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_hometown_position_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_hometown_position_count.columns = [key1, key2, 'hometown_position_count']

        df_hometown_position_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_hometown_position_conversion_rate.columns = [key1, key2, 'hometown_position_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_hometown_position_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_hometown_position_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['hometown_position_count'].fillna(0, inplace=True)
    df_train['hometown_position_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['hometown_position_count', 'hometown_position_conversion_rate']]
    df_save.columns = ['hometown_position_count', 'hometown_position_conversion_rate']
    savePath = "hometown_position.csv"
    df_save.to_csv(savePath, index=False, header=True)


# positionID_telecomsOperator
def position_telecomsOperator_count(concat_data):
    key1 = 'positionID'
    key2 = 'telecomsOperator'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_position_telecomsOperator_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_position_telecomsOperator_count.columns = [key1, key2, 'position_telecomsOperator_count']

        df_position_telecomsOperator_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_position_telecomsOperator_conversion_rate.columns = [key1, key2, 'position_telecomsOperator_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_position_telecomsOperator_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_position_telecomsOperator_conversion_rate, how='left', on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['position_telecomsOperator_count'].fillna(0, inplace=True)
    df_train['position_telecomsOperator_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['position_telecomsOperator_count', 'position_telecomsOperator_conversion_rate']]
    df_save.columns = ['position_telecomsOperator_count', 'position_telecomsOperator_conversion_rate']
    savePath = "position_telecomsOperator.csv"
    df_save.to_csv(savePath, index=False, header=True)


# creativeID_positionID
def creativeID_positionID_count(concat_data):
    key1 = 'creativeID'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_creativeID_positionID_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_creativeID_positionID_count.columns = [key1, key2, 'creativeID_positionID_count']

        df_creativeID_positionID_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_creativeID_positionID_conversion_rate.columns = [key1, key2, 'creativeID_positionID_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_creativeID_positionID_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_creativeID_positionID_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['creativeID_positionID_count'].fillna(0, inplace=True)
    df_train['creativeID_positionID_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['creativeID_positionID_count', 'creativeID_positionID_conversion_rate']]
    df_save.columns = ['creativeID_positionID_count', 'creativeID_positionID_conversion_rate']
    savePath = "creativeID_positionID.csv"
    df_save.to_csv(savePath, index=False, header=True)


# education_positionID
def education_positionID(concat_data):
    key1 = 'education'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_education_positionID_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_education_positionID_count.columns = [key1, key2, 'education_positionID_count']

        df_education_positionID_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_education_positionID_conversion_rate.columns = [key1, key2, 'education_positionID_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_education_positionID_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_education_positionID_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['education_positionID_count'].fillna(0, inplace=True)
    df_train['education_positionID_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['education_positionID_count', 'education_positionID_conversion_rate']]
    df_save.columns = ['education_positionID_count', 'education_positionID_conversion_rate']
    savePath = "education_positionID.csv"
    df_save.to_csv(savePath, index=False, header=True)


# campaignID_positionID
def campaignID_positionID_count(concat_data):
    key1 = 'camgaignID'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_campaignID_positionID_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_campaignID_positionID_count.columns = [key1, key2, 'campaignID_positionID_count']

        df_campaignID_positionID_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_campaignID_positionID_conversion_rate.columns = [key1, key2, 'campaignID_positionID_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_campaignID_positionID_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_campaignID_positionID_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['campaignID_positionID_count'].fillna(0, inplace=True)
    df_train['campaignID_positionID_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['campaignID_positionID_count', 'campaignID_positionID_conversion_rate']]
    df_save.columns = ['campaignID_positionID_count', 'campaignID_positionID_conversion_rate']
    savePath = "campaignID_positionID.csv"
    df_save.to_csv(savePath, index=False, header=True)


# marriageStatus_positionID
def marriageStatus_positionID_count(concat_data):
    key1 = 'marriageStatus'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_marriageStatus_positionID_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_marriageStatus_positionID_count.columns = [key1, key2, 'marriageStatus_positionID_count']

        df_marriageStatus_positionID_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_marriageStatus_positionID_conversion_rate.columns = [key1, key2, 'marriageStatus_positionID_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_marriageStatus_positionID_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_marriageStatus_positionID_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['marriageStatus_positionID_count'].fillna(0, inplace=True)
    df_train['marriageStatus_positionID_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['marriageStatus_positionID_count', 'marriageStatus_positionID_conversion_rate']]
    df_save.columns = ['marriageStatus_positionID_count', 'marriageStatus_positionID_conversion_rate']
    savePath = "marriageStatus_positionID.csv"
    df_save.to_csv(savePath, index=False, header=True)


# residence_positionID
def residence_positionID_count(concat_data):
    key1 = 'residence'
    key2 = 'positionID'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_residence_positionID_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_residence_positionID_count.columns = [key1, key2, 'residence_positionID_count']

        df_residence_positionID_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_residence_positionID_conversion_rate.columns = [key1, key2, 'residence_positionID_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_residence_positionID_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_residence_positionID_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['residence_positionID_count'].fillna(0, inplace=True)
    df_train['residence_positionID_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['residence_positionID_count', 'residence_positionID_conversion_rate']]
    df_save.columns = ['residence_positionID_count', 'residence_positionID_conversion_rate']
    savePath = "residence_positionID.csv"
    df_save.to_csv(savePath, index=False, header=True)


# age_telecomsOperator
def age_telecomsOperator_count(concat_data):
    key1 = 'age'
    key2 = 'telecomsOperator'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_age_telecomsOperator_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_age_telecomsOperator_count.columns = [key1, key2, 'age_telecomsOperator_count']

        df_age_telecomsOperator_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_age_telecomsOperator_conversion_rate.columns = [key1, key2, 'age_telecomsOperator_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_age_telecomsOperator_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_age_telecomsOperator_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['age_telecomsOperator_count'].fillna(0, inplace=True)
    df_train['age_telecomsOperator_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['age_telecomsOperator_count', 'age_telecomsOperator_conversion_rate']]
    df_save.columns = ['age_telecomsOperator_count', 'age_telecomsOperator_conversion_rate']
    savePath = "age_telecomsOperator.csv"
    df_save.to_csv(savePath, index=False, header=True)


# age_education
def age_education_count(concat_data):
    key1 = 'age'
    key2 = 'education'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_age_education_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_age_education_count.columns = [key1, key2, 'age_education_count']

        df_age_education_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_age_education_conversion_rate.columns = [key1, key2, 'age_education_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_age_education_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_age_education_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['age_education_count'].fillna(0, inplace=True)
    df_train['age_education_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['age_education_count', 'age_education_conversion_rate']]
    df_save.columns = ['age_education_count', 'age_education_conversion_rate']
    savePath = "age_education.csv"
    df_save.to_csv(savePath, index=False, header=True)


# camgaignID_connectionType
def camgaignID_connectionType_count(concat_data):
    key1 = 'camgaignID'
    key2 = 'connectionType'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_camgaignID_connectionType_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_camgaignID_connectionType_count.columns = [key1, key2, 'camgaignID_connectionType_count']

        df_camgaignID_connectionType_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_camgaignID_connectionType_conversion_rate.columns = [key1, key2, 'camgaignID_connectionType_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_camgaignID_connectionType_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_camgaignID_connectionType_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['camgaignID_connectionType_count'].fillna(0, inplace=True)
    df_train['camgaignID_connectionType_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['camgaignID_connectionType_count', 'camgaignID_connectionType_conversion_rate']]
    df_save.columns = ['camgaignID_connectionType_count', 'camgaignID_connectionType_conversion_rate']
    savePath = "camgaignID_connectionType.csv"
    df_save.to_csv(savePath, index=False, header=True)


# gender_education
def gender_education_count(concat_data):
    key1 = 'gender'
    key2 = 'education'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_gender_education_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_gender_education_count.columns = [key1, key2, 'gender_education_count']

        df_gender_education_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_gender_education_conversion_rate.columns = [key1, key2, 'gender_education_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_gender_education_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_gender_education_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['gender_education_count'].fillna(0, inplace=True)
    df_train['gender_education_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['gender_education_count', 'gender_education_conversion_rate']]
    df_save.columns = ['gender_education_count', 'gender_education_conversion_rate']
    savePath = "gender_education.csv"
    df_save.to_csv(savePath, index=False, header=True)


# marriageStatus_residence
def marriageStatus_residence_count(concat_data):
    key1 = 'marriageStatus'
    key2 = 'residence'
    window = 50000
    df_train = pd.DataFrame()
    for day in range(17, 27):
        day_start = day * 10000
        day_end = day_start + window
        temp_data = concat_data[(concat_data['clickTime'] >= day_start) & (concat_data['clickTime'] < day_end)]

        df_marriageStatus_residence_count = temp_data.groupby([key1, key2]).size().reset_index()
        df_marriageStatus_residence_count.columns = [key1, key2, 'marriageStatus_residence_count']

        df_marriageStatus_residence_conversion_rate = temp_data.groupby([key1, key2]).apply(
            lambda df: np.mean(df['label'])).reset_index()
        df_marriageStatus_residence_conversion_rate.columns = [key1, key2, 'marriageStatus_residence_conversion_rate']

        day_start_train = day_end
        day_end_train = day_start_train + 10000
        temp_data_train = concat_data[
            (concat_data['clickTime'] >= day_start_train) & (
                concat_data['clickTime'] < day_end_train)]
        temp_data_train = pd.merge(temp_data_train, df_marriageStatus_residence_count, how='left', on=[key1, key2])
        temp_data_train = pd.merge(temp_data_train, df_marriageStatus_residence_conversion_rate, how='left',
                                   on=[key1, key2])
        df_train = pd.concat([df_train, temp_data_train], axis=0, ignore_index=True)

    df_train['marriageStatus_residence_count'].fillna(0, inplace=True)
    df_train['marriageStatus_residence_conversion_rate'].fillna(0, inplace=True)
    df_save = df_train[['marriageStatus_residence_count', 'marriageStatus_residence_conversion_rate']]
    df_save.columns = ['marriageStatus_residence_count', 'marriageStatus_residence_conversion_rate']
    savePath = "marriageStatus_residence.csv"
    df_save.to_csv(savePath, index=False, header=True)


if __name__ == "__main__":
    pass

