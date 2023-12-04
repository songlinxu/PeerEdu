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
from utils import normalize_table_by_video, visual_pearson_matrix, logistic_regression_all, _get_group_id



def f2():
    result_file = 'dataset/gaze_aoi_max.csv'
    user_table_select = pd.read_csv(result_file) 
    user_table_select = normalize_table_by_video(user_table_select,'percent')

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        
    datatype = 'percent'
    sns.stripplot(data=user_table_select, x="group_name", y=datatype,dodge=False, alpha=.5, legend=False,ax=axes[0],palette=['lightblue', 'lightgreen'])
    sns.boxplot(data=user_table_select, x="group_name", y=datatype, ax=axes[0],palette=['lightblue', 'lightgreen'], width=0.4,showfliers=False)
    axes[0].set_ylabel(datatype, fontsize=16)

    axes[0].tick_params(axis='x', labelsize=16)  # Set x-axis tick font size
    axes[0].tick_params(axis='y', labelsize=16)  # Set y-axis tick font size
        

    result_file = 'dataset/question_result_test.csv'
    user_table_select = pd.read_csv(result_file)  
    user_table_select = normalize_table_by_video(user_table_select,'test_score_avg')

    datatype = 'test_score_avg'
    sns.stripplot(data=user_table_select, x="group_name", y=datatype,dodge=False, alpha=.5, legend=False,ax=axes[1],palette=['lightblue', 'lightgreen'])
    sns.boxplot(data=user_table_select, x="group_name", y=datatype, ax=axes[1],palette=['lightblue', 'lightgreen'], width=0.4,showfliers=False)
    
    axes[1].set_ylabel('Post Test Accuracy', fontsize=16)

    axes[1].tick_params(axis='x', labelsize=16)  # Set x-axis tick font size
    axes[1].tick_params(axis='y', labelsize=16)  # Set y-axis tick font size

    plt.show() 

def f3():
    result_file = 'dataset/question_result_test.csv'
    user_table_select = pd.read_csv(result_file)  
    group_mapping = {'control': 0, 'feedback': 1}
    user_table_select['group_id'] = user_table_select['group_name'].map(group_mapping)

    # pearson_matrix
    datatype_list = ['group_id','test_score_easy','test_score_hard','test_score_avg','tlx_score','confusion_dur','inattention_dur','fall_num_avg','valid_percent','follow_percent']
    visual_pearson_matrix(user_table_select,datatype_list,True)



def f4():
    result_file = 'dataset/question_result_each_test.csv'
    user_table_select = pd.read_csv(result_file)  
    user_table_select['group_id'] = user_table_select['group_name'].apply(_get_group_id)
    user_table_select = user_table_select[user_table_select['accuracy']!=-1]

    logistic_regression_all(user_table_select,True) 


