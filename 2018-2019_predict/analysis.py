# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:06:11 2022

@author: maths
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

mode = 'per_game'

csv_filename_train = 'all_nba_player_{mode}_train.csv'.format(mode=mode)
csv_filename_test = 'all_nba_player_{mode}_test.csv'.format(mode=mode)

features = {
    'Year' : 0,
    'Player':1,
    'Pos':2,
    'Age':3,
    'Tm':4, #team
    'G':5,  #Games played
    'GS':6,
    'MP':7, # minutes per game
    'FG':8,
    'FGA':9,
    'FG%':10,
    '3P':11,
    '3PA':12,
    '3P%':13,
    '2P':14,
    '2PA':15,
    '2P%':16,
    'eFG%':17,
    'FT':18,
    'FTA':19,
    'FT%':20,
    'ORB':21,
    'DRB':22,
    'TRB':23,
    'AST':24,
    'STL':25,
    'BLK':26,
    'TOV':27,
    'PF':28,
    'PTS':29,
    'all-nba_1':30,
    'all-nba_2':31,
    'all-nba_3':32,
    'all-defensive_1':33,
    'all-defensive_2':34,
    'all-rookie_1':35,
    'all-rookie_2':36,
    'all_star_game_rosters_1':37,
    'all_star_game_rosters_2':38,
    'class':39
  }

features_list = ['Age','G','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
def read_data(csv_file):

    
    df = pd.read_csv(csv_file)
    #print(df.head())
    X1=df.iloc[:,features['Age']] 
  
    X3=df.iloc[:,features['G']]
    X4=df.iloc[:,features['MP']]
    X5=df.iloc[:,features['FG']]
    X6=df.iloc[:,features['FGA']]
    X7=df.iloc[:,features['3P']]
    X8=df.iloc[:,features['3PA']]
    X9=df.iloc[:,features['2P']]
    X10=df.iloc[:,features['2PA']]
    X11=df.iloc[:,features['FT']]
    X12=df.iloc[:,features['FTA']]
    X13=df.iloc[:,features['ORB']]
    X14=df.iloc[:,features['DRB']]
    X15=df.iloc[:,features['TRB']]
    X16=df.iloc[:,features['AST']]
    X17=df.iloc[:,features['STL']]
    X18=df.iloc[:,features['BLK']]
    X19=df.iloc[:,features['TOV']]
    X20=df.iloc[:,features['PF']]
    X21=df.iloc[:,features['PTS']]
    X = np.column_stack((X1,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21))
    
    y = df.iloc[:,features['class']]
    return X,y


def norm(X):
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X_sclaed = min_max_scaler.fit_transform(X)
    # df = pd.DataFrame(x_scaled)
    
    # X_sclaed = StandardScaler().fit_transform(X)
    
    return X_sclaed

binary_class_problem = 1
def reduce_class(y):
    # class 0 for nothing
    # class 1 for all-nba
    # class 2 for all-defender
    if(binary_class_problem):
        y[y==1] = 1 #set rookie 1st to 0
        y[y==2] = 1 #set rookie 1st to 0
        y[y==3] = 1 #set rookie 2nd to 0
        
        y[y==4] = 0 #set rookie 1st to 0
        y[y==5] = 0 #set rookie 2nd to 0
        
        y[y==6] = 0 #set rookie 1st to 0
        y[y==7] = 0 #set rookie 2nd to 0
    else:
        pass
    
    return y

def select_feature(X,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel

 
    x_df = pd.DataFrame(
        data=X, 
        columns=features_list)

    selector = SelectFromModel(estimator=LogisticRegression()).fit(x_df, y)
  
    print(selector.estimator_.coef_)
    #coef_ is of shape (1, n_features)
    importance = np.abs(selector.estimator_.coef_[0])
    feature_names = np.array(features_list)
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importances via coefficients")
    plt.xticks(rotation=90)
    plt.show()
    
    
    print(selector.estimator_.coef_)
    #coef_ is of shape (1, n_features)
    efs = selector.estimator_.coef_[0]
    importance = np.abs(efs[efs >1.0])
    feature_names = np.array(features_list)
    plt.bar(height=importance, x=feature_names[efs >1.0])
    plt.title("Selected Features")
    plt.xticks(rotation=90)
    plt.show()
    
    
    feature_idx = selector.get_support()
    feature_name = x_df.columns[feature_idx]
    
    print(feature_name)
    # model.get_feature_names_out()
    X_new = selector.transform(X)
    # print(X_new.shape, X_new)
def change_class_to_label(y):
    if binary_class_problem:
        y[y==0] = 'No Award'
        y[y==1] = 'all-nba'
    else:
        y[y==0] = 'No Award'
        y[y==1] = 'all-nba-1st'
        y[y==2] = 'all-nba-2nd'
        y[y==3] = 'all-nba-3rd'
        y[y==4] = 'all-defender-1nd'
        y[y==5] = 'all-defender-2nd'
        y[y==6] = 'all-rookie-1'
        y[y==7] = 'all-rookie-2'
    return y

def data_vis_multiclass(X, y):
    import matplotlib.pyplot as plt
    import seaborn as sns

    y = change_class_to_label(y)
    y = np.column_stack((y,))

    print(type(X), type(y))
    print(X.shape, y.shape)
    x_df = pd.DataFrame(
        data=X, 
        columns=features_list)

    x_df_tag = pd.DataFrame(
        data = y,
        columns =['class'])
    x_df = pd.concat([x_df, x_df_tag] , axis=1)
    
    
    # sns.FacetGrid(x_df, hue="class", size=5).map(plt.scatter, "PCA1", "PCA2").add_legend()
    sns.FacetGrid(x_df, hue="class", size=5).map(plt.scatter, "DRB", "AST", s=2).add_legend()
    plt.title('nba data from 1980-2022: binary') 
    
    plt.show()

def data_vis_multiclass_pca(X, y):
    import matplotlib.pyplot as plt
    import seaborn as sns

    y = change_class_to_label(y)
    y = np.column_stack((y,))

    print(type(X), type(y))
    print(X.shape, y.shape)
    x_df = pd.DataFrame(
        data=X, 
        columns=['PCA1', 'PCA2'])

    x_df_tag = pd.DataFrame(
        data = y,
        columns =['class'])
    x_df = pd.concat([x_df, x_df_tag] , axis=1)
    
    # sns.FacetGrid(x_df, hue="class", size=5).map(plt.scatter, "AST", "PTS").add_legend()
    sns.FacetGrid(x_df, hue="class", size=5).map(plt.scatter, "PCA1", "PCA2",s=2).add_legend()
    if binary_class_problem:
        plt.title('nba data from 1980-2022: binary') 
    else:
        plt.title('nba data from 1980-2022: multinomial') 
    plt.show()
    

def pca_analysis(X_train):
    # how many features we need
    pca = PCA(n_components=0.99)
     
    # Fit and transform data
    pca.fit_transform(X_train)
     
    # Bar plot of explained_variance
    plt.bar(
        range(1,len(pca.explained_variance_)+1),
        pca.explained_variance_
        )
      
    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.show()
    
def data_summary(y):
    y = y.tolist()
    print('size of data is',len(y))
    print('size of no label is', y.count(0))
    print('size of 1 label is', y.count(1))
    print('size of 2 label is', y.count(2))
    print('size of 3 label is', y.count(3))
    print('size of 4 label is', y.count(4))
    print('size of 5 label is', y.count(5))
    print('size of 6 label is', y.count(6))
    print('size of 7 label is', y.count(7))
    
    pass

def ref():
    import numpy as np
    import matplotlib.pyplot as plt
      
    N = 3
    ind = np.arange(N) 
    width = 0.25
      
    xvals = [8, 9, 2]
    bar1 = plt.bar(ind, xvals, width, color = 'r')
      
    yvals = [10, 20, 30]
    bar2 = plt.bar(ind+width, yvals, width, color='g')
      
    zvals = [11, 12, 13]
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'b')
      
    plt.xlabel("Dates")
    plt.ylabel('Scores')
    plt.title("Players Score")
      
    plt.xticks(ind+width,['2021Feb01', '2021Feb02', '2021Feb03'])
    plt.legend( (bar1, bar2, bar3), ('Player1', 'Player2', 'Player3') )
    plt.show()

def plt_pred_summary():
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    labels = ['lOGlOSS', 'Precision', 'Recall', 'F1score']
    SVC = [0.039, 0.77, 0.57, 0.65]
    KNN = [0.041, 0.80, 0.40, 0.53]
    RF = [0.051, 0.71, 0.40, 0.51]
    MLP = [0.052, 0.76, 0.43, 0.55]
    DUMMY = [0.773, 0.00, 0.00, 0.0] 
    

    N = 4
    ind = np.arange(N) 
    width = 0.1
      
   
    bar1 = plt.bar(ind, SVC, width, color = 'r')  
    bar2 = plt.bar(ind+width, KNN, width, color='g')   
    bar3 = plt.bar(ind+width*2, RF, width, color = 'b')
    bar4 = plt.bar(ind+width*3, MLP, width, color = 'y')
    bar5 = plt.bar(ind+width*4, DUMMY, width, color = 'c')
    
    plt.xlabel("merics")
    plt.ylabel('Scores')
    plt.title("scores by different aproches")
      
    plt.xticks(ind+width, labels)
    plt.legend( (bar1, bar2, bar3, bar4, bar5), ('SVC', 'KNN', 'RF', 'MLP', 'DUMMY'),loc = 'upper left' )
    plt.show()


    
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
 
    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    X_train, y_train = read_data(csv_filename_train)
    X_test, y_test = read_data(csv_filename_test)
    import pandas
    data_summary(y_train.append(y_test))
    # data_summary(y_test)
    
    reduce_class(y_train)
    reduce_class(y_test)
    data_vis_multiclass(X_train, y_train)
 
    print('-----------------',type(y_test))
# must do norm for svm, otherwise program stuck, why?
    X_train = norm(X_train)
    X_test = norm(X_test)
    
    pca_analysis(X_train)
     
    select_feature(X_train, y_train)
    
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)

    data_vis_multiclass_pca(X_train, y_train)
    # ref()
    plt_pred_summary()
    
     
