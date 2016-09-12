from util import cross_validate

import pandas
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score

dataset = pandas.read_csv('bezdekIris.csv',
                          header=None,
                          names=['sepal_length', 'sepal_width', 'petal_length',
                                 'petal_width', 'class'])[:-1]

# SVM
clf = SVC(kernel='linear')
svm_scores = cross_val_score(clf, dataset[dataset.columns[:-1]],
                             dataset[dataset.columns[-1]], cv=5)

a, b = cross_validate(clf, dataset[dataset.columns[:-1]].as_matrix(), dataset[dataset.columns[-1]].as_matrix(), n_folds=5)
print sum(a) / len(a)
print sum(b) / len(b)

print 'SVM Accuracy: %0.2f (+/- %0.2f)' % (svm_scores.mean(),
                                           svm_scores.std() * 2)



# ANN
from sknn.mlp import Classifier, Layer

# df = pandas.concat([dataset, pandas.get_dummies(dataset['class'])], axis=1)

X_train = dataset[dataset.columns[:-1]].as_matrix()
# X_train = Normalizer().transform(X_train)
y_train = dataset[dataset.columns[-1]].as_matrix()

nn = Classifier(
    layers=[
        Layer('Rectifier', units=5),
        Layer('Softmax')],
    learning_rate=0.02, n_iter=150)

a, b = cross_validate(nn, X_train, y_train, n_folds=5, stratified=True)
print sum(a) / len(a)
print sum(b) / len(b)

nn.fit(X_train, y_train)
