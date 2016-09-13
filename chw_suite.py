from chw_data import CHWData

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
    learning_rate=0.02, n_iter=25, verbose=False)

estimators = {'Decision Tree': tree, 'Random Forest': forest,
              'SVM': svm, 'Neural Network': nn}

for name, estimator in estimators.iteritems():
    scores = cross_val_score(estimator, chw_data.features_m,
                             chw_data.targets_m, cv=5)
    print '%s Accuracy: %0.2f (+/- %0.2f)' % (name, scores.mean(),
                                              scores.std() * 2)
