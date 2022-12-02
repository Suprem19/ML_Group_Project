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
from plotnine import *

mode = 'per_game'

# csv_filename = 'all_nba_player_{mode}.csv'.format(mode=mode)

csv_filename_train = '../doc/all_nba_player_{mode}_train.csv'.format(mode=mode)
csv_filename_test = '../doc/all_nba_player_{mode}_test.csv'.format(mode=mode)

features = {
    'Year': 0,
    'Player': 1,
    'Pos': 2,
    'Age': 3,
    'Tm': 4,  # team
    'G': 5,  # Games played
    'GS': 6,
    'MP': 7,  # minutes per game
    'FG': 8,
    'FGA': 9,
    'FG%': 10,
    '3P': 11,
    '3PA': 12,
    '3P%': 13,
    '2P': 14,
    '2PA': 15,
    '2P%': 16,
    'eFG%': 17,
    'FT': 18,
    'FTA': 19,
    'FT%': 20,
    'ORB': 21,
    'DRB': 22,
    'TRB': 23,
    'AST': 24,
    'STL': 25,
    'BLK': 26,
    'TOV': 27,
    'PF': 28,
    'PTS': 29,
    'all-nba_1': 30,
    'all-nba_2': 31,
    'all-nba_3': 32,
    'all-defensive_1': 33,
    'all-defensive_2': 34,
    'all-rookie_1': 35,
    'all-rookie_2': 36,
    'all_star_game_rosters_1': 37,
    'all_star_game_rosters_2': 38,
    'class': 39
}

# confusion matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

csv_list = []


def plt_cm(y_true, y_predict, model_name, labels=[-1, 1]):
    cm = confusion_matrix(y_test, y_predict, labels=labels)
    # print(est.classes_,type(est.classes_))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    # disp.ax_.set_title("Add a nice title here after calling disp.plot()")
    disp.ax_.set_title('Confusion Matrix of %s' % model_name)
    plt.show()


def knn(X_train, X_test, y_train, y_test):
    recall = []
    F1 = []
    precision = []
    train_score_list = []
    test_score_list = []
    for k in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        train_score_list.append(knn.score(X_train, y_train))
        test_score_list.append(knn.score(X_test, y_test))
        precision.append(metrics.precision_score(y_test, y_pred))
        F1.append(metrics.f1_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))

    # plt.plot(range(1, 41), train_score_list, label='train_score')
    # plt.plot(range(1, 41), test_score_list, label='test_score')
    # plt.xlabel('k_value')
    # plt.ylabel('socre')
    # plt.legend(loc='best')
    # plt.title('Performance of Different K-value on Train and Test Score')
    # plt.show()
    #
    plt.plot(range(1, 41), precision, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', label='presision')
    plt.plot(range(1, 41), recall, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='green', label='recall')
    plt.plot(range(1, 41), F1, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='yellow', label='f1_score')

    plt.xlabel('k_value')
    plt.ylabel('socre')
    plt.legend(loc='best')
    plt.title('Performance of Different K-value on Precision, Recall and F1_score')
    plt.show()

    # k = np.argmax(test_score_list) + 1
    knn = KNeighborsClassifier(n_neighbors=27, weights='distance')
    knn.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # print('------------k nearest: ', k)
    print(metrics.classification_report(y_test, y_pred))
    # metrics.f1_score(y_test,y_pred)
    return knn


def mlp(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    score = []
    recall = []
    precision = []
    F1 = []
    for i in range(1, 31):
        cnn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=i, random_state=999, activation='relu')
        cnn.fit(X_train, y_train.values.ravel())
        y_pred = cnn.predict(X_test)
        score.append(metrics.accuracy_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        F1.append(metrics.f1_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))
    plt.plot(range(1, 31), score, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='c', label='accuracy_score')
    plt.plot(range(1, 31), precision, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', label='presision')
    plt.plot(range(1, 31), recall, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='green', label='recall')
    plt.plot(range(1, 31), F1, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='yellow', label='f1_score')
    plt.xlabel('layers')
    plt.ylabel('values')
    plt.legend(loc='best')
    plt.title('Value of Accuracy,Recall,Precision and F1 Score with MLP')
    plt.show()
    cnn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=18, random_state=999, activation='relu')
    cnn.fit(X_train, y_train.values.ravel())
    y_pred = cnn.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    return cnn


def rf(X_train, X_test, y_train, y_test):
    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    recall = []
    precision = []
    F1 = []
    for i in range(1, 21):
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=i)
        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        recall.append(metrics.recall_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        F1.append(metrics.f1_score(y_test, y_pred))
    plt.plot(range(1, 21), precision, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', label='presision')
    plt.plot(range(1, 21), recall, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='green', label='recall')
    plt.plot(range(1, 21), F1, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='yellow', label='f1_score')
    plt.xlabel('value of min_sample_leaf')
    plt.ylabel('values')
    plt.legend(loc='best')
    plt.title('Value of Accuracy,Recall,Precision and F1 Score with RF')
    plt.show()
    print(metrics.classification_report(y_test, y_pred))

    return clf


'''Classification metrics can't handle a mix of multiclass and continuous targets
'''


def dummy(X_train, X_test, y_train, y_test):
    from sklearn.dummy import DummyClassifier
    # load the dataset
    # define model
    model = DummyClassifier(strategy="most_frequent")
    # fit model
    model.fit(X_train, y_train)
    # define new data
    # make a prediction
    y_pred = model.predict(X_test)
    # summarize prediction
    print(metrics.classification_report(y_test, y_pred))
    return model

def read_data(csv_file):
    df = pd.read_csv(csv_file)
    # print(df.head())
    # X1=df.iloc[:,features['Age']]

    X3 = df.iloc[:, features['G']]
    # X4=df.iloc[:,features['MP']]
    X5 = df.iloc[:, features['FG']]
    # X6=df.iloc[:,features['FGA']]
    X7 = df.iloc[:, features['3P']]
    # X8=df.iloc[:,features['3PA']]
    X9 = df.iloc[:, features['2P']]
    # X10=df.iloc[:,features['2PA']]
    X11 = df.iloc[:, features['FT']]
    X12 = df.iloc[:, features['FTA']]
    # X13=df.iloc[:,features['ORB']]
    X14 = df.iloc[:, features['DRB']]
    X15 = df.iloc[:, features['TRB']]
    X16 = df.iloc[:, features['AST']]
    # X17=df.iloc[:,features['STL']]
    # X18=df.iloc[:,features['BLK']]
    # X19=df.iloc[:,features['TOV']]
    X20 = df.iloc[:, features['PF']]
    X21 = df.iloc[:, features['PTS']]
    # X = np.column_stack((X1,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21))
    X = np.column_stack((X3, X5, X7, X9, X11, X12, X14, X15, X16, X20, X21))

    y = df.iloc[:, features['class']]
    return X, y


class_weights_enabled = 1
threshold_svm_enabled = 1
norm_svm_enabled = 1


def svm(X_train, X_test, y_train, y_test, class_weight={0: 1, 1: 5}):
    import matplotlib.pyplot as plt
    from sklearn import svm

    # fit the model and get the separating hyperplane using weighted classes
    if class_weights_enabled:
        wclf = svm.SVC(kernel="linear", probability=True, class_weight=class_weight)
    else:
        wclf = svm.SVC(kernel="linear", probability=True)
    # wclf = svm.SVC(kernel="linear")
    wclf.fit(X_train, y_train)
    y_dec = wclf.decision_function(X_test)
    y_predic_prop = wclf.predict_proba(X_test)
    y_pred = wclf.predict(X_test)

    if threshold_svm_enabled:
        y_pred = (wclf.predict_proba(X_test)[:, 1] >= 0.1284).astype(bool)
    # y_pred = predict_post(y_predic_prop, y_pred)

    m = metrics.classification_report(y_test, y_pred)
    print(m)
    precicion_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    print(f1_score)
    # return (precicion_score, recall_score, f1_score)
    return wclf


def norm(X):
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X_sclaed = min_max_scaler.fit_transform(X)
    # df = pd.DataFrame(x_scaled)

    # X_sclaed = StandardScaler().fit_transform(X)

    return X_sclaed


def pca(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test


def reduce_class(y):
    # class 0 for nothing
    # class 1 for all-nba
    # class 2 for all-defender

    y[y == 1] = 1  # set all-nba 1st to 0
    y[y == 2] = 1  # set all-nba 2nd to 0
    y[y == 3] = 1  # set all-nba 3rd to 0

    y[y == 4] = 0  # set all-defender 1st to 0
    y[y == 5] = 0  # set all-defender 2nd to 0

    y[y == 6] = 0  # set rookie 1st to 0
    y[y == 7] = 0  # set rookie 2nd to 0

    return y


def read_csv(csv_filename):
    import csv
    with open(csv_filename, encoding='UTF8', newline='') as f:
        csv_rd = csv.reader(f, delimiter=',')
        for row in csv_rd:
            csv_list.append(row)
    f.close()


# https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293#:~:text=ROC%20curve%20for%20finding%20the,positives%20and%20100%25%20true%20positives.
def thres_hold_choose(X_train, X_test, y_train, y_test):
    from sklearn.metrics import roc_curve  # Calculate the ROC curve
    from sklearn.metrics import precision_recall_curve  # Calculate the Precision-Recall curve
    from sklearn import svm
    # Fit the model
    wclf = svm.SVC(kernel="linear", probability=True, class_weight={0: 1, 1: 2.45})
    wclf.fit(X_train, y_train)
    # Predict the probabilities
    y_pred = wclf.predict_proba(X_test)
    # Get the probabilities for positive class
    y_pred = y_pred[:, 1]

    # Import module for data visualization

    import plotnine

    # Create the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot the ROC curve
    df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
    print(df_fpr_tpr.head())

    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))

    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    # Create data viz
    plotnine.options.figure_size = (8, 4.8)

    viz = (ggplot(data=df_fpr_tpr) +
           geom_point(aes(x='FPR',
                          y='TPR'),
                      size=0.4) +
           # Best threshold
           geom_point(aes(x=fprOpt,
                          y=tprOpt),
                      color='#981220',
                      size=4) +
           geom_line(aes(x='FPR',
                         y='TPR')) +
           geom_text(aes(x=fprOpt,
                         y=tprOpt),
                     label='Optimal threshold \n for class: {}'.format(thresholdOpt),
                     nudge_x=0.14,
                     nudge_y=-0.10,
                     size=10,
                     fontstyle='italic') +
           labs(title='ROC Curve') +
           xlab('False Positive Rate (FPR)') +
           ylab('True Positive Rate (TPR)') +
           theme_minimal())
    print(viz)


def select_3(pred_list, pos, pred):
    max_prob = 0
    player = ''
    index = 0
    cat = []
    for i, item in enumerate(pred_list):
        if (item[1] == pos):
            cat.append(item + [i])
    cat = sorted(cat, key=lambda cat: cat[2], reverse=True)
    cat_3 = cat[:4]
    print('final reulst----------------------------------')
    print(cat_3)
    for r in cat_3:
        pred[r[3]] = 1


def predict_post(pred_prop, pred):
    all_nba_class = 1

    pred_refined = np.zeros(pred.shape)
    print(type(pred), pred.shape)

    predic_trues, = np.where(pred == all_nba_class)
    print(type(predic_trues), predic_trues)

    pred_list = []
    for i in predic_trues:
        row = csv_list[i]
        # print(row[features['Player']], row[features['Pos']], pred_prop[:,1][i])
        pred_list.append([row[features['Player']], row[features['Pos']], pred_prop[:, 1][i]])

    print(pred_list)
    # 1st is player, 2nd pos, 3rd propability
    # print(pred_list[pred_list[1]=='C'])
    select_3(pred_list, 'C', pred_refined)
    select_3(pred_list, 'PF', pred_refined)
    select_3(pred_list, 'SG', pred_refined)
    select_3(pred_list, 'PG', pred_refined)
    select_3(pred_list, 'SG', pred_refined)
    return pred_refined


def plot_weighs(a):
    plt.plot(a[:, 0], a[:, 1], color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', label='presision')
    plt.plot(a[:, 0], a[:, 2], color='blue', linestyle='dashed',
             marker='o', markerfacecolor='green', label='recall')
    plt.plot(a[:, 0], a[:, 3], color='blue', linestyle='dashed',
             marker='o', markerfacecolor='yellow', label='f1_score')

    plt.legend(loc='lower left')
    plt.xlabel('Weights of class1 (all-naba)')
    plt.ylabel('Scores')
    plt.title('Scores for different value of class_weight')
    plt.show()


def svm_weight_try(X_train, X_test, y_train, y_test):
    weights = np.linspace(2.0, 5.0, num=20, endpoint=False)
    print('SVM---------------------')
    weights_info = []
    for w in weights:
        print('weights is: ', w)
        (precicion_score, recall_score, f1_score) = svm(X_train, X_test, y_train, y_test, {0: 1, 1: w})
        weights_info.append([w, precicion_score, recall_score, f1_score])

    weights_np = np.array(weights_info)
    print(weights_np)


def roc_curve(model, X_test, y_test):
    proba = model.predict_proba(X_test)
    pos_prob = proba[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, pos_prob)

    return fpr, tpr, pos_prob


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    read_csv(csv_filename_test)
    X_train, y_train = read_data(csv_filename_train)
    X_test, y_test = read_data(csv_filename_test)

    if norm_svm_enabled:
        # must do norm for svm, otherwise program stuck, why?
        X_train = norm(X_train)
        X_test = norm(X_test)

    # X_train, X_test = pca(X_train, X_test)

    reduce_class(y_train)
    reduce_class(y_test)

    # y1= df.iloc[:,features['all-nba_1']]
    # y2= df.iloc[:,features['all-nba_2']]
    # y3= df.iloc[:,features['all-nba_3']]
    # y4= df.iloc[:,features['all-defensive_1']]
    # y5= df.iloc[:,features['all-defensive_2']]
    # y = np.column_stack((y1,y2,y3,y4,y5))
    thres_hold_choose(X_train, X_test, y_train, y_test)

    # convert class vectors to binary class matrices
    # y = keras.utils.to_categorical(y, num_classes)
    # print('rondom forest---------------------')
    # rf = rf(X_train, X_test, y_train, y_test)
    #
    # print('KNN---------------------')
    # knn = knn(X_train, X_test, y_train, y_test)

    print('ANN---------------------')
    mlp(X_train, X_test, y_train, y_test)
    # print('SVM---------------------')
    # svm = svm(X_train, X_test, y_train, y_test, {0: 1, 1: 2.45})


    print('DUMMY---------------------')
    dummy = dummy(X_train, X_test, y_train, y_test)

    plt.style.use('fivethirtyeight')

    roc, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)

    fpr, tpr, pos_prob = roc_curve(svm, X_test, y_test)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.set_title("SVC: %.2f" % metrics.roc_auc_score(y_test, pos_prob), size=15, x=.485, ha='center')

    fpr, tpr, pos_prob = roc_curve(rf, X_test, y_test)
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_title("RF: %.2f" % metrics.roc_auc_score(y_test, pos_prob), size=15, x=.485, ha='center')

    fpr, tpr, pos_prob = roc_curve(knn, X_test, y_test)
    ax3.plot(fpr, tpr)
    ax3.plot([0, 1], [0, 1], linestyle='--')
    ax3.set_title("KNN: %.2f" % metrics.roc_auc_score(y_test, pos_prob), size=15, x=.485, ha='center')

    fpr, tpr, pos_prob = roc_curve(mlp, X_test, y_test)
    ax4.plot(fpr, tpr)
    ax4.plot([0, 1], [0, 1], linestyle='--')
    ax4.set_title("DNN: %.2f" % metrics.roc_auc_score(y_test, pos_prob), size=15, x=.485, ha='center')

    roc.suptitle("Model ROC Curves", y=1.045, weight='bold', size=18)

    roc.savefig('../results/roc.png', dpi=400, bbox_inches='tight')

    fpr, tpr, pos_prob = roc_curve(dummy, X_test, y_test)
    plt.plot(fpr, tpr)
    plt.title("Dummy: %.2f" % metrics.roc_auc_score(y_test, pos_prob), size=15, x=.485, ha='center')
    plt.show()

    proba1 = svm.predict_proba(X_test)
    print("Log loss: %.3f" % metrics.log_loss(y_test, proba1))

    proba2 = knn.predict_proba(X_test)
    print("Log loss: %.3f" % metrics.log_loss(y_test, proba2))

    proba3 = rf.predict_proba(X_test)
    print("Log loss: %.3f" % metrics.log_loss(y_test, proba3))

    proba4 = mlp.predict_proba(X_test)
    print("Log loss: %.3f" % metrics.log_loss(y_test, proba4))

    proba5 = dummy.predict_proba(X_test)
    print("Log loss: %.3f" % metrics.log_loss(y_test, proba5))

