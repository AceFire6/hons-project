from warnings import filterwarnings
from sklearn.cross_validation import KFold, StratifiedKFold


def cross_validate(estimator, x_data, y_data, n_folds=3, stratified=False):
    if stratified:
        kfold = StratifiedKFold(y_data, n_folds=n_folds)
    else:
        kfold = KFold(len(y_data), n_folds=n_folds)

    test_scores = []
    train_scores = []
    for train, test in kfold:
        x_test, x_train = x_data[test], x_data[train]
        y_test, y_train = y_data[test], y_data[train]

        estimator.fit(x_train, y_train)
        train_score = estimator.score(x_train, y_train)
        train_scores.append(train_score)

        test_score = estimator.score(x_test, y_test)
        test_scores.append(test_score)

    return train_scores, test_scores


def filter_warnings():
    # Preserve output sanity.
    # These warnings don't affect anything and are unnecessary.
    filterwarnings('ignore', 'numpy not_equal will not check object')
    filterwarnings('ignore', 'downsample module has been moved to')
