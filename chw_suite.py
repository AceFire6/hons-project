import json

from chw_data import CHWData

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sknn.platform import gpu64
from sknn.mlp import Classifier, Layer


matplotlib.rcParams['backend'] = "Qt4Agg"
label = 'activeQ2'
unused_features = ['projectCode', 'userCode']
categorical_features = ['country', 'sector']

chw_data = CHWData('chw_data.csv', label,
                   unused_features, categorical_features)

tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
svm = SVC()
nn = Classifier(
    layers=[
        Layer('Rectifier', units=10),
        Layer('Softmax')],
    learning_rate=0.02, n_iter=150, verbose=None)

estimators = {'Decision_Tree': tree, 'Random_Forest': forest,
              'SVM': svm, 'Neural_Network': nn}


def param_run(num_x, cross_folds, drop_cols=list()):
    results = []
    for estimator_name, estimator in estimators.iteritems():
        print '[%d] Starting %s' % (num_x, estimator_name)
        feature_data = chw_data.get_features(num_x).drop(drop_cols)
        feature_data = (feature_data.as_matrix()
                        if estimator_name == 'Neural Network'
                        else feature_data)
        cross_score = cross_val_score(estimator, n_jobs=-1,
                                      X=feature_data,
                                      y=chw_data.targets, cv=cross_folds)
        print '     %s Accuracy: %0.2f (+/- %0.2f)' % (estimator_name,
                                                       cross_score.mean(),
                                                       cross_score.std() * 2)
        results.append((estimator_name, cross_score))
    return results

all_scores = {i: [] for i in estimators.keys()}
# Go through all values of X (1-90)
x_val_range = range(1, 91)
for x in x_val_range:
    result_scores = param_run(x, cross_folds=10)
    for result in result_scores:
        name, val = result
        all_scores[name].append(val.mean())
        with open('x-vals-%s-scores.json' % name, 'a+') as fout:
            fout.write(json.dumps(val.tolist()) + '\n')

legend = []
styles = iter(['-', '--', '-.', ':'])
plt.ioff()
for name, scores in all_scores.iteritems():
    plt.plot(x_val_range, scores, next(styles))
    legend.append(name)

plt.legend(legend, loc=4)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Number of days included')
plt.savefig('x-vals-compare.png')
