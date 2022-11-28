import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from operator import itemgetter
from sklearn.model_selection import cross_val_score

# 导入数据集

dfHistorical = pd.read_csv('historical-all-nba.csv')
dfCurrent = pd.read_csv('current-all-nba.csv')
dfHistorical.head()

# 数据可视化
all_nba = dfHistorical.loc[dfHistorical['All-NBA'] == 1]
non_all_nba = dfHistorical.loc[dfHistorical['All-NBA'] == 0]

# 场均得分PPG可视化
plt.style.use('fivethirtyeight')

ppg_wins, ax = plt.subplots()

ax.scatter(all_nba['PTS'], all_nba['Team Wins'], label="All-NBA players", s=25)
ax.scatter(non_all_nba['PTS'], non_all_nba['Team Wins'], label="The rest", s=25)

ax.legend(loc='best', prop={'size': 9, 'weight': 'normal', "family": "Times New Roman"})

ax.set_xlabel('PPG')
ax.set_ylabel('Team Wins')

ppg_wins.suptitle("PPG, Wins, and All-NBA Selections", weight='bold', size=18)

ppg_wins.savefig('ppg_wins.png', dpi=400, bbox_inches='tight')

# 正负值WS可视化
plt.style.use('fivethirtyeight')

ppg_ws, ax = plt.subplots()

ax.scatter(all_nba['PTS'], all_nba['WS'], label="All-NBA players", s=25)
ax.scatter(non_all_nba['PTS'], non_all_nba['WS'], label="The rest", s=25)

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('PPG')
ax.set_ylabel('WS')

ppg_ws.suptitle("PPG, WS, and All-NBA Selections", weight='bold', size=18)

ppg_ws.savefig('ppg_ws.png', dpi=400, bbox_inches='tight')

# 不可替代值vorp可视化
vorp_ws, ax = plt.subplots()

ax.scatter(all_nba['VORP'], all_nba['WS'], label="All-NBA players", s=25)
ax.scatter(non_all_nba['VORP'], non_all_nba['WS'], label="The rest", s=25)

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('VORP')
ax.set_ylabel('WS')

vorp_ws.suptitle("VORP, WS, and All-NBA Selections", weight='bold', size=18)

vorp_ws.savefig('vorp_ws.png', dpi=400, bbox_inches='tight')

# 四个直方图：与 PPG、VORP 和 WS 之间的差异相比，
# 最佳阵容 和非最佳阵容球员在球队获胜方面的差异很小。
# WS似乎显示出两组球员之间最大的不同。

# PPG直方图
ppg_hist, ax = plt.subplots()

ax.hist(all_nba['PTS'], alpha=.75, label='All-NBA players')
ax.hist(non_all_nba['PTS'], alpha=.75, label='The rest')

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('PPG')
ax.set_ylabel('Frequency')

ppg_hist.suptitle("PPG Histogram", weight='bold', size=18)

ppg_hist.savefig('ppg_hist.png', dpi=400, bbox_inches='tight')

# VORP直方图
vorp_hist, ax = plt.subplots()

ax.hist(all_nba['VORP'], alpha=.75, label='All-NBA players')
ax.hist(non_all_nba['VORP'], alpha=.75, label='The rest')

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('VORP')
ax.set_ylabel('Frequency')

vorp_hist.suptitle("VORP Histogram", weight='bold', size=18)

vorp_hist.savefig('vorp_hist.png', dpi=400, bbox_inches='tight')

# WS直方图
ws_hist, ax = plt.subplots()

ax.hist(all_nba['WS'], alpha=.75, label='All-NBA players')
ax.hist(non_all_nba['WS'], alpha=.75, label='The rest')

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('WS')
ax.set_ylabel('Frequency')

ws_hist.suptitle("WS Histogram", weight='bold', size=18)

ws_hist.savefig('ws_hist.png', dpi=400, bbox_inches='tight')

# wins直方图
wins_hist, ax = plt.subplots()

ax.hist(all_nba['Team Wins'], alpha=.75, label='All-NBA players')
ax.hist(non_all_nba['Team Wins'], alpha=.75, label='The rest')

ax.legend(loc='best', prop={'size': 9, "family": "Times New Roman"})

ax.set_xlabel('Team Wins')
ax.set_ylabel('Frequency')

wins_hist.suptitle("Team Wins Histogram", weight='bold', size=18)

wins_hist.savefig('wins_hist.png', dpi=400, bbox_inches='tight')

train, test = train_test_split(dfHistorical, test_size=0.25, random_state=36)

# 模型创建
xtrain = train[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'VORP', 'WS', 'All-Star']]
ytrain = train[['All-NBA']]

xtest = test[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'VORP', 'WS', 'All-Star']]
ytest = test[['All-NBA']]

print("Training set size: %.0f" % len(xtrain))
print("Testing set size: %.0f" % len(xtest))


# 准确率，召回率，预测值，F1值，交叉验证
# 交叉验证分(k = 3) 和这些分数的 95% 置信区间
def scores(model):
    model.fit(xtrain, ytrain.values.ravel())
    y_pred = model.predict(xtest)

    print("Accuracy score: %.3f" % metrics.accuracy_score(ytest, y_pred))
    print("Recall: %.3f" % metrics.recall_score(ytest, y_pred))
    print("Precision: %.3f" % metrics.precision_score(ytest, y_pred))
    print("F1: %.3f" % metrics.f1_score(ytest, y_pred))

    proba = model.predict_proba(xtest)
    print("Log loss: %.3f" % metrics.log_loss(ytest, proba))

    pos_prob = proba[:, 1]
    print("Area under ROC curve: %.3f" % metrics.roc_auc_score(ytest, pos_prob))

    cv = cross_val_score(model, xtest, ytest.values.ravel(), cv=3, scoring='accuracy')
    print("Accuracy (cross validation score): %0.3f (+/- %0.3f)" % (cv.mean(), cv.std() * 2))

    return y_pred


svc = SVC(kernel='rbf', gamma=1e-3, C=100, probability=True)
y_svc = scores(svc)

rf = RandomForestClassifier(random_state=999, n_estimators=100, criterion='gini')
y_rf = scores(rf)

knn = neighbors.KNeighborsClassifier(n_neighbors=12, weights='uniform')
y_knn = scores(knn)

dnn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=100, random_state=999, activation='relu')
y_dnn = scores(dnn)


# Confusion matrix: 可视化模型的准确性
#  true positives （右下）、 true negatives（左上）、
# false positives（右上）和 false negatives（左下）

def confusion_matrix(y_pred, model_name):
    cm = metrics.confusion_matrix(ytest, y_pred)

    plt.style.use("fivethirtyeight")
    z, ax = plt.subplots()

    sns.heatmap(cm, annot=True, ax=ax, linewidth=2, fmt='g')

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    z.suptitle("%s Confusion Matrix" % model_name.upper(), weight='bold', size=18, x=.45)

    z.savefig('%s_cm.png' % model_name, dpi=400, bbox_inches='tight')


confusion_matrix(y_svc, 'svc')
confusion_matrix(y_rf, 'rf')
confusion_matrix(y_knn, 'knn')
confusion_matrix(y_dnn, 'dnn')


# 画ROC曲线
def roc_curve(model):
    proba = model.predict_proba(xtest)
    pos_prob = proba[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(ytest, pos_prob)

    return fpr, tpr, pos_prob


plt.style.use('fivethirtyeight')

roc, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)

fpr, tpr, pos_prob = roc_curve(svc)
ax1.plot(fpr, tpr)
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.set_title("SVC: %.2f" % metrics.roc_auc_score(ytest, pos_prob), size=15, x=.485, ha='center')

fpr, tpr, pos_prob = roc_curve(rf)
ax2.plot(fpr, tpr)
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_title("RF: %.2f" % metrics.roc_auc_score(ytest, pos_prob), size=15, x=.485, ha='center')

fpr, tpr, pos_prob = roc_curve(knn)
ax3.plot(fpr, tpr)
ax3.plot([0, 1], [0, 1], linestyle='--')
ax3.set_title("KNN: %.2f" % metrics.roc_auc_score(ytest, pos_prob), size=15, x=.485, ha='center')

fpr, tpr, pos_prob = roc_curve(dnn)
ax4.plot(fpr, tpr)
ax4.plot([0, 1], [0, 1], linestyle='--')
ax4.set_title("DNN: %.2f" % metrics.roc_auc_score(ytest, pos_prob), size=15, x=.485, ha='center')

roc.suptitle("Model ROC Curves", y=1.045, weight='bold', size=18)

roc.savefig('roc.png', dpi=400, bbox_inches='tight')

# 预测
dfCurrentNames = dfCurrent.iloc[:, 0]
dfCurrentPredict = dfCurrent[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'VORP', 'WS', 'All-Star']]

dfCurrent.head()


def make_pred(model):
    proba = model.predict_proba(dfCurrentPredict)
    pos_prob = proba[:, 1]

    combined_list = [[i, j] for i, j in zip(dfCurrentNames, pos_prob)]
    combined_list = sorted(combined_list, key=itemgetter(1), reverse=True)

    for i in combined_list:
        print(i)

    return pos_prob


svc_prob = make_pred(svc)
rf_prob = make_pred(rf)
knn_prob = make_pred(knn)
dnn_prob = make_pred(dnn)

avg_prob = []

for i, j, k, l in zip(svc_prob, rf_prob, knn_prob, dnn_prob):
    avg_prob.append((i + j + k + l) / 4)

avg_list = [[i, j] for i, j in zip(dfCurrentNames, avg_prob)]
avg_list = sorted(avg_list, key=itemgetter(1), reverse=True)

for i in avg_list:
    print(i)
