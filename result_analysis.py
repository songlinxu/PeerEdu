import os, json, math 
import pandas as pd 
import numpy as np 
from scipy.stats import ttest_ind
import seaborn as sns 
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import webvtt, json, cv2  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils import normalize_table_by_video, calculate_ANOVA, visual_box, visual_distribution, calculate_mean_std, visual_stack_bar_feedback_score, _get_group_id, visual_pearson_matrix, logistic_regression_all

def s1_gaze_manipulate():
    # 1. Valid Focus:  ============================================= 
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file) 
    # calculate_mean_std(user_table_select,'valid_percent')
    # user_table_select = normalize_table_by_video(user_table_select,'valid_percent')
    # visual_distribution(user_table_select, 'valid_percent')
    # calculate_ANOVA(user_table_select, 'valid_percent')
    # visual_box(user_table_select, 'valid_percent','box')

    # 1. Course Following:  ============================================= 
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)
    # calculate_mean_std(user_table_select,'follow_percent')
    # user_table_select = normalize_table_by_video(user_table_select,'follow_percent')
    # visual_distribution(user_table_select, 'follow_percent')
    # calculate_ANOVA(user_table_select, 'follow_percent')
    # visual_box(user_table_select, 'follow_percent','box')
    


    # 1. Gaze manipulation: fall_num_avg ============================================= 
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file) 
    # calculate_mean_std(user_table_select,'fall_num_avg')
    # user_table_select = normalize_table_by_video(user_table_select,'fall_num_avg')
    # visual_distribution(user_table_select, 'fall_num_avg')
    # calculate_ANOVA(user_table_select, 'fall_num_avg')
    # visual_box(user_table_select, 'fall_num_avg','point')
    

    # 1. Gaze manipulation: gaze_aoi_max =============================================
    result_file = 'dataset/gaze_aoi_max.csv'
    user_table_select = pd.read_csv(result_file) 
    calculate_mean_std(user_table_select,'percent')
    user_table_select = normalize_table_by_video(user_table_select,'percent')
    visual_distribution(user_table_select, 'percent')
    calculate_ANOVA(user_table_select, 'percent')
    visual_box(user_table_select, 'percent','box')

def s2_learn_experience():
    # 1. Learning experience: inattention =============================================
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)  
    # calculate_mean_std(user_table_select,'inattention_dur')
    # user_table_select = normalize_table_by_video(user_table_select,'inattention_dur')
    # visual_distribution(user_table_select, 'inattention_dur')
    # calculate_ANOVA(user_table_select, 'inattention_dur')
    # visual_box(user_table_select, 'inattention_dur','point')

    # 1. Learning experience: confusion =============================================
    result_file = 'dataset/question_result_test.csv'
    user_table_select = pd.read_csv(result_file)  
    # -- Use zscore threshold to remove outliers (This is because rare participants' confusion duration value is very extreme)
    z_scores = stats.zscore(user_table_select['confusion_dur'])
    z_threshold = 3
    user_table_select = user_table_select[(z_scores < z_threshold) & (z_scores > -z_threshold)]
    calculate_mean_std(user_table_select,'confusion_dur')
    user_table_norm = normalize_table_by_video(user_table_select,'confusion_dur')
    visual_distribution(user_table_select, 'confusion_dur')
    calculate_ANOVA(user_table_norm, 'confusion_dur')
    visual_box(user_table_norm, 'confusion_dur','point')

    # 1. Learning experience: NASA TLX score =============================================
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)  
    # calculate_mean_std(user_table_select,'tlx_score')
    # user_table_norm = normalize_table_by_video(user_table_select,'tlx_score')
    # visual_distribution(user_table_select, 'tlx_score')
    # calculate_ANOVA(user_table_norm, 'tlx_score')
    # visual_box(user_table_norm, 'tlx_score','point')

    # 1. Learning experience: feedback score =============================================
    # result_file = 'dataset/question_result_test.csv'
    # visual_stack_bar_feedback_score(result_file)


def s3_learn_outcome():
    # 2. Learning outcome: test score =============================================
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)  
    # calculate_mean_std(user_table_select,'test_score_easy')
    # user_table_select = normalize_table_by_video(user_table_select,'test_score_easy')
    # visual_distribution(user_table_select, 'test_score_easy')
    # calculate_ANOVA(user_table_select, 'test_score_easy')
    # visual_box(user_table_select, 'test_score_easy','point')

    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)  
    # calculate_mean_std(user_table_select,'test_score_hard')
    # user_table_select = normalize_table_by_video(user_table_select,'test_score_hard')
    # # visual_distribution(user_table_select, 'test_score_hard')
    # calculate_ANOVA(user_table_select, 'test_score_hard')
    # visual_box(user_table_select, 'test_score_hard','point')

    result_file = 'dataset/question_result_test.csv'
    user_table_select = pd.read_csv(result_file)  
    calculate_mean_std(user_table_select,'test_score_avg')
    user_table_select = normalize_table_by_video(user_table_select,'test_score_avg')
    visual_distribution(user_table_select, 'test_score_avg')
    calculate_ANOVA(user_table_select, 'test_score_avg')
    visual_box(user_table_select, 'test_score_avg','box','Post Test Accuracy')


def s4_decode_learn():
    # 4.1 . Decoding student learning behaviors: average decoding =============================================
    # result_file = 'dataset/question_result_test.csv'
    # user_table_select = pd.read_csv(result_file)  
    # group_mapping = {'control': 0, 'feedback': 1}
    # user_table_select['group_id'] = user_table_select['group_name'].map(group_mapping)
    # # pearson_matrix
    # datatype_list = ['group_id','test_score_easy','test_score_hard','test_score_avg','tlx_score','confusion_dur','inattention_dur','fall_num_avg','valid_percent','follow_percent']
    # visual_pearson_matrix(user_table_select,datatype_list,True)

    # 4.2 . Decoding student learning behaviors: individual decoding =============================================
    # Important Warning: we must filter accuracy = -1

    result_file = 'dataset/question_result_each_test.csv'
    user_table_select = pd.read_csv(result_file)  
    user_table_select['group_id'] = user_table_select['group_name'].apply(_get_group_id)
    user_table_select = user_table_select[user_table_select['accuracy']!=-1]

    logistic_regression_all(user_table_select,True) 



s1_gaze_manipulate()



