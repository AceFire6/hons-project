from chw_data import CHWData

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


unused_features = ['projectCode', 'userCode']
label = 'activeQ2'
categorical_features = ['country', 'sector']

chw_data = CHWData('chw_data.csv', label,
                   unused_features, categorical_features)

forest = RandomForestClassifier()

x_train, x_test, y_train, y_test = chw_data.get_test_train_data_m(0.8)
forest.fit(x_train, y_train)
print forest.score(x_test, y_test)

scores = cross_val_score(forest, chw_data.features_m, chw_data.targets_m, cv=3)

print 'Decision Tree Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
                                                     scores.std() * 2)
