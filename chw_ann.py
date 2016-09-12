from chw_data import CHWData

from sklearn.cross_validation import cross_val_score
from sknn.mlp import Classifier, Layer

unused_features = ['projectCode', 'userCode']
label = 'activeQ2'
categorical_features = ['country', 'sector']

chw_data = CHWData('chw_data.csv', label,
                   unused_features, categorical_features)

nn = Classifier(
    layers=[
        Layer('Sigmoid', units=5),
        Layer('Softmax')],
    learning_rate=0.02,
    n_iter=15, verbose=True, debug=True)

scores = cross_val_score(nn, chw_data.features, chw_data.targets, cv=5)

print 'ANN Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
                                           scores.std() * 2)
