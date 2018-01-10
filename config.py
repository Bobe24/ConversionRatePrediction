#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Author: Bobe_24@126.com

train_path = "train.csv"
# train_1_path = "train_1.csv"
# train_2_path = "train_2.csv"
test_path = "test.csv"
user_path = "user.csv"
ad_path = "ad.csv"
app_categories_path = "app_categories.csv"
position_path = "position.csv"
user_app_action_path = "user_app_actions.csv"
user_installedapps_path = "user_installedapps.csv"
train_duplicate_path = "trainFeature-4.txt"
test_duplicate_path = "testFeature-4.txt"
app_actions_count_path = "app_actions_count_past16days.csv"
app_installed_count_path = "app_installed_count.csv"
user_actions_count_path = "user_actions_count_past16days.csv"
user_installed_count_path = "user_installed_count.csv"
train_cor_path = "train-cor.csv"
test_cor_path = "test-cor.csv"
train_td_path = "TD-train.csv"
test_td_path = "TD-test.csv"
position_connectionType = "position_connectionType.csv"
advertiser_position = "advertiser_position.csv"
gender_position = "gender_position.csv"
hometown_residence = "hometown_residence.csv"
age_marriageStatus = "age_marriageStatus.csv"
age_position = "age_position.csv"
app_position = "app_position.csv"
hometown_position = "hometown_position.csv"
position_telecomsOperator = "position_telecomsOperator.csv"
creativeID_positionID = "creativeID_positionID.csv"
education_positionID_path = "education_positionID.csv"
campaignID_positionID = "campaignID_positionID.csv"
marriageStatus_positionID = "marriageStatus_positionID.csv"
residence_positionID = "residence_positionID.csv"
age_telecomsOperator = "age_telecomsOperator.csv"
age_education = "age_education.csv"
camgaignID_connectionType = "camgaignID_connectionType.csv"
gender_education = "gender_education.csv"
marriageStatus_residence = "marriageStatus_residence.csv"

enc_clm = ['gender', 'education', 'marriageStatus', 'haveBaby', 'sitesetID', 'positionType',
           'connectionType', 'telecomsOperator', 'clickWeek', 'ageClass', 'age_gender', 'appID',
           'appCategory', 'appPlatform']

numer_col_names = ['clickHour']
