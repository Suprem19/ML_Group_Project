import numpy as np
import pandas as pd

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

mode = 'per_game'


# csv_filename = 'all_nba_player_{mode}.csv'.format(mode=mode)

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

#confusion matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plt_cm(y_true, y_predict, model_name, labels=[-1,1]):
    cm = confusion_matrix(y_test, y_predict, labels=labels)
    #print(est.classes_,type(est.classes_))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    #disp.ax_.set_title("Add a nice title here after calling disp.plot()")
    disp.ax_.set_title('Confusion Matrix of %s'%model_name)
    plt.show()
  
def knn(X_train, X_test, y_train, y_tes):
    for i in range(1,2):
        #Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
        
        #Train the model using the training sets
        knn.fit(X_train, y_train)
        
        #Predict the response for test dataset
        y_pred = knn.predict(X_test)
        
        print('------------k nearest: ', i)
        print(metrics.classification_report(y_test, y_pred))
        # metrics.f1_score(y_test,y_pred)    

def mlp(X_train, X_test, y_train, y_tes):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(5,2),
                        max_iter = 1000,
                        solver = 'adam')

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

def rf(X_train, X_test, y_train, y_tes):
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

'''Classification metrics can't handle a mix of multiclass and continuous targets
'''
def ridge(X_train, X_test, y_train, y_tes):
    from sklearn.linear_model import Ridge
    # load the dataset
    # define model
    model = Ridge(alpha=1.0)
    # fit model
    model.fit(X_train, y_train)
    # define new data
    # make a prediction
    y_pred = model.predict(X_test)
    # summarize prediction
    print(metrics.classification_report(y_test, y_pred))

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

if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    X_train, y_train = read_data(csv_filename_train)
    X_test, y_test = read_data(csv_filename_test)
    # y1= df.iloc[:,features['all-nba_1']]
    # y2= df.iloc[:,features['all-nba_2']]
    # y3= df.iloc[:,features['all-nba_3']]
    # y4= df.iloc[:,features['all-defensive_1']]
    # y5= df.iloc[:,features['all-defensive_2']]
    # y = np.column_stack((y1,y2,y3,y4,y5))
    
 
    # convert class vectors to binary class matrices
    # y = keras.utils.to_categorical(y, num_classes)
    print('rondom forest---------------------')
    rf(X_train, X_test, y_train, y_test)
    
    print('KNN---------------------')
    knn(X_train, X_test, y_train, y_test)
  
    print('MLP---------------------')
    mlp(X_train, X_test, y_train, y_test)
   