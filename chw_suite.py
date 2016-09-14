from chw_data import CHWData

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sknn.mlp import Classifier, Layer

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
    learning_rate=0.02, n_iter=20, verbose=False)

estimators = {'Decision Tree': tree, 'Random Forest': forest,
              'SVM': svm, 'Neural Network': nn}


def param_run(num_x, cross_folds):
    results = []
    for name, estimator in estimators.iteritems():
        scores = cross_val_score(estimator, chw_data.get_features(num_x, True),
                                 chw_data.targets_m,
                                 cv=cross_folds)
        print '[%d] %s Accuracy: %0.2f (+/- %0.2f)' % (num_x, name,
                                                       scores.mean(),
                                                       scores.std() * 2)
        results.append((name, scores.mean()))
    return results


all_scores = {i: [] for i in estimators.keys()}
# Go through all values of X (1-90)
x_val_range = range(1, 91)
for x in x_val_range:
    results = param_run(x, cross_folds=2)
    for result in results:
        name, val = result
        all_scores[name].append(val)

legend = []
for name, scores in all_scores.iteritems():
    plt.plot(x_val_range, scores)
    legend.append(name)

plt.legend(legend)
plt.show()

