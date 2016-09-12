from chw_data import CHWData

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, train_test_split


unused_features = ['projectCode', 'userCode']
label = 'activeQ2'
categorical_features = ['country', 'sector']

chw_data = CHWData('chw_data.csv', label,
                   unused_features, categorical_features)

svm = SVC()
test_train_data = train_test_split(chw_data.features, chw_data.targets,
                                   train_size=0.8, random_state=5)
x_train, x_test, y_train, y_test = [i.as_matrix() for i in test_train_data]
svm.fit(x_train, y_train)
print svm.score(x_test, y_test)

scores = cross_val_score(svm, X=chw_data.features_m,
                         y=chw_data.targets_m, cv=3)

print 'SVM Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
                                           scores.std() * 2)
