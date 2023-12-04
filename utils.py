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



def calculate_mean_std(user_table_select, datatype):
    print('\n\n\nanalyzing: ',datatype)
    for video in [1,2,3,4,5]:
        data_item = user_table_select[(user_table_select['video_id']==video)]
        print('video: ', video)
        control_group = data_item[data_item['group_name']=='control']
        feedback_group = data_item[data_item['group_name']=='feedback']
        print('control: mean: ',np.mean(control_group[datatype]),' std: ',np.std(control_group[datatype]))
        print('feedback: mean: ',np.mean(feedback_group[datatype]),' std: ',np.std(feedback_group[datatype]))
        print('improve percent: ',(np.mean(feedback_group[datatype])-np.mean(control_group[datatype]))/np.mean(control_group[datatype]))
        print('\n')

    control_group = user_table_select[user_table_select['group_name']=='control']
    feedback_group = user_table_select[user_table_select['group_name']=='feedback']
    print('control: mean: ',np.mean(control_group[datatype]),' std: ',np.std(control_group[datatype]))
    print('feedback: mean: ',np.mean(feedback_group[datatype]),' std: ',np.std(feedback_group[datatype]))
    print('improve percent: ',(np.mean(feedback_group[datatype])-np.mean(control_group[datatype]))/np.mean(control_group[datatype]))
    print('\n\n\n')


def calculate_ANOVA(user_table_select, datatype):
    print('analyzing: ',datatype)
    for video in [1,2,3,4,5]:
        data_item = user_table_select[(user_table_select['video_id']==video)]
        
        model = ols(datatype+' ~ C(group_name)',data=data_item).fit()
        result = sm.stats.anova_lm(model, type=3)
        print('video: ', video)
        print(result)
        control_group = data_item[data_item['group_name']=='control']
        feedback_group = data_item[data_item['group_name']=='feedback']
        print('control: mean: ',np.mean(control_group[datatype]),' std: ',np.std(control_group[datatype]))
        print('feedback: mean: ',np.mean(feedback_group[datatype]),' std: ',np.std(feedback_group[datatype]))


    model = ols(datatype+' ~ C(video_id) + C(group_name) + C(video_id):C(group_name)',data=user_table_select).fit()
    result = sm.stats.anova_lm(model, type=3)
    print(result)
    control_group = user_table_select[user_table_select['group_name']=='control']
    feedback_group = user_table_select[user_table_select['group_name']=='feedback']
    print('control: mean: ',np.mean(control_group[datatype]),' std: ',np.std(control_group[datatype]))
    print('feedback: mean: ',np.mean(feedback_group[datatype]),' std: ',np.std(feedback_group[datatype]))

   

    

def normalize_table_by_video(user_table_select,datatype):
    grouped_table = user_table_select.groupby('video_id')

    def normalize_group(group):
        result_values = group[datatype].values.reshape(-1, 1)
        scaler = StandardScaler()
        normalized_result = scaler.fit_transform(result_values)
        group[datatype] = normalized_result
        return group

    normalized_table = grouped_table.apply(normalize_group)

    normalized_table = normalized_table.reset_index(drop=True)

    return normalized_table



def visual_box(user_table_select, datatype, plot_type, ylabel = None):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    video_list = [1,2,3,4,5]
    group_list = ['control','feedback']
    if plot_type == 'box':
        sns.boxplot(user_table_select,x='video_id',y=datatype,hue='group_name',ax=axes[0])
        sns.boxplot(user_table_select,x='group_name',y=datatype,ax=axes[1])

    elif plot_type == 'bar':
        sns.barplot(user_table_select,x='video_id',y=datatype,hue='group_name',errorbar=('ci', 95),ax=axes[0])
        sns.barplot(user_table_select,x='group_name',y=datatype,errorbar=('ci', 95),ax=axes[1])

    elif plot_type == 'point':
        sns.pointplot(user_table_select,x='video_id',y=datatype,hue='group_name',errorbar=('ci', 95),ax=axes[0],markers=["o", "s"], linestyles=["-", "--"])
        sns.pointplot(user_table_select,x='group_name',y=datatype,errorbar=('ci', 95),ax=axes[1])

    sns.stripplot(data=user_table_select, x="video_id", y=datatype, hue="group_name",dodge=True, alpha=.2, legend=False,ax=axes[2])
    sns.pointplot(data=user_table_select, x="video_id", y=datatype, hue="group_name",dodge=.4, linestyle="none", errorbar=None,marker="_", markersize=20, markeredgewidth=3,ax=axes[2])
        
    sns.stripplot(data=user_table_select, x="group_name", y=datatype,dodge=False, alpha=.5, legend=False,ax=axes[3],palette=['lightblue', 'lightgreen'])
    sns.boxplot(data=user_table_select, x="group_name", y=datatype, ax=axes[3],palette=['lightblue', 'lightgreen'], width=0.5,showfliers=False)
    
    if ylabel == None:
        axes[3].set_ylabel(datatype)
    else:
        axes[3].set_ylabel(ylabel)

    
    plt.show() 

def visual_stack_bar_feedback_score(data_file):
    df = pd.read_csv(data_file)
    df = df[(df['group_name']=='feedback')]
    
    result = []
    
    for i in range(8):
        count_dict = {1:0,2:0,3:0,4:0,5:0}
        for v,video_id in enumerate([1,2,3,4,5]):
            df_video = df[(df['video_id']==video_id)]
            student_id_list = list(set(df_video['student_id']))
            for j,student_id in enumerate(student_id_list):
                data_item = df_video[df_video['student_id']==student_id]
                assert len(data_item) == 1
                count_dict[data_item['feedback_q'+str(i+1)].values[0]] += 1
        result.append(['q'+str(i+1),count_dict[1],count_dict[2],count_dict[3],count_dict[4],count_dict[5]])
    result = pd.DataFrame(np.array(result),columns=['feedback_question','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree'])
    result['Strongly Disagree'] = result['Strongly Disagree'].astype(float)
    result['Disagree'] = result['Disagree'].astype(float)
    result['Neutral'] = result['Neutral'].astype(float)
    result['Agree'] = result['Agree'].astype(float)
    result['Strongly Agree'] = result['Strongly Agree'].astype(float)
    print(result)
    result.plot(x = 'feedback_question', kind = 'barh', stacked = True, title = 'Stacked Bar Graph', mark_right = True) 
    plt.show()

def visual_pearson_matrix(user_table_select,datatype_list,norm=False):
    if norm == True:
        for target_item in datatype_list:
            if target_item not in ['group_id','video_id']:
                user_table_select = normalize_table_by_video(user_table_select,target_item)
    factor_dict =  {'group_id': 'Group ID','test_score_easy': 'Easy Test Score','test_score_hard': 'Hard Test Score','test_score_avg': 'All Test Score','tlx_score': 'Cognitive Workload','confusion_dur': 'Confusion Duration','inattention_dur': 'Inattention Duration','fall_num_avg': 'AOI-falled Gaze','valid_percent': 'Valid Focus','follow_percent': 'Course Following'}
    data = {}
    for datatype in datatype_list:
        data[factor_dict[datatype]] = np.array(user_table_select[datatype])

    df = pd.DataFrame(data)

    correlation_matrix = df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='Blues', fmt=".2f", vmax=.8, center=0,square=True, linewidths=3, annot_kws={"fontsize":12},cbar_kws={"shrink": .5, 'location': 'right', 'orientation': 'vertical', 'pad': 0})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # plt.title('Pearson Correlation Matrix', fontsize=24)
    plt.tight_layout()
    # plt.savefig('corr.pdf')
    plt.show()


def _get_question_id_val(raw_str):
    return int(raw_str.split('q')[1])

def logistic_regression_all(data_table,norm=False):
    data_table['question_id_val'] = data_table['question_id'].apply(_get_question_id_val)
    target_list = ['group_id','inattention_dur','confusion_dur','fall_num_avg','valid_percent','follow_percent']
    if norm == True:
        for target_item in target_list:
            if target_item not in ['group_id','video_id']:
                data_table = normalize_table_by_video(data_table,target_item)
    group_mapping = {'control': 0, 'feedback': 1}
    data_table['group_id'] = data_table['group_name'].map(group_mapping)
    X = data_table[target_list]
    y = data_table['accuracy']

    model2 = sm.Logit(y, X)
    try:
        result = model2.fit()
    except: 
        pass
    p_values = result.pvalues

    print(result.summary())
    print(result.params)
    coef_list = list(result.params.values)
    p_values_list = [round(p_val,4) for p_val in p_values]

    print('coefficients: ',coef_list)
    print('p_values_list: ',p_values_list)

    plt.bar(X.columns, coef_list, color='b', alpha=0.5, width=0.2)
    for i, v in enumerate(p_values_list):
        # plt.text(i, coef_list[i], 'P = '+str(v), ha="center", va="bottom")
        if v < 0.001:
            plt.text(i, coef_list[i], '***', ha="center", va="bottom")
        elif v < 0.01:
            plt.text(i, coef_list[i], '**', ha="center", va="bottom")
        elif v < 0.05:
            plt.text(i, coef_list[i], '*', ha="center", va="bottom")
    plt.xlabel('Independent Variables')
    plt.ylabel('Coefficient Value',fontsize=16)
    plt.ylim(-0.2,1)
    plt.xticks(np.arange(0, 6), ['Group ID','Inattention Duration','Confusion Duration','Gaze Fall Ratio','Valid Focus','Course Following'],fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    # print(p_values['fall_num_avg'])


def _get_group_id(raw_value):
    if raw_value == 'control': 
        return 0
    else:
        return 1


def visual_distribution(data_table,datatype):
    fig, axes = plt.subplots(1, 8, figsize=(14, 2))
    sns.kdeplot(data_table, x=datatype, color="blue", ax=axes[0])
    for v in [1,2,3,4,5]:
        data_item = data_table[data_table['video_id']==v]
        sns.kdeplot(data_item, x=datatype, color="blue", ax=axes[v])
    
    sns.kdeplot(data_table, x=datatype, hue='group_name', ax=axes[6])
    sns.kdeplot(data_table, x=datatype, hue='video_id', ax=axes[7])
    plt.show()
