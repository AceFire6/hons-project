from util import filter_warnings

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, normalize


class CHWData(object):
    def __init__(self, file_name, label='', drop_cols=list(),
                 categorical_features=list()):
        filter_warnings()
        self._file_name = file_name
        self._dataset = DataFrame.from_csv(file_name)
        self._features = self._dataset.columns
        self._processed_dataset = None
        self.drop_cols = drop_cols
        self.categorical_cols = categorical_features
        self.label = label
        if label:
            self._process_dataset(label, True, drop_cols, categorical_features)

    def _process_dataset(self, label, assign=True, drop_cols=list(),
                         categorical_features=list()):
        dataset = self._dataset
        dataset.drop(drop_cols, axis=1, inplace=True)

        features = dataset.columns.drop(label)
        label_encoder = LabelEncoder()

        for feature in categorical_features:
            dataset[feature] = label_encoder.fit_transform(dataset[feature])

        if categorical_features:
            number_features = features.difference(categorical_features)
            dataset[number_features] = normalize(dataset[number_features])

        if assign:
            self._processed_dataset = dataset
            self._features = features

        return dataset

    def update(self, label='', drop_cols=list(), categorical_features=list()):
        self._process_dataset(label, True, drop_cols, categorical_features)

    @property
    def dataset(self):
        if self._processed_dataset is not None:
            return self._processed_dataset
        return self._dataset

    @property
    def feature_labels(self):
        return self._features

    @property
    def features(self):
        return self.dataset[self._features]

    @property
    def features_m(self):
        return self.features.as_matrix()

    def get_features(self, x_num=90, as_matrix=False):
        x_labels = self._get_x_labels(x_num)
        feats = self.features.drop(x_labels, axis=1)
        if as_matrix:
            return feats.as_matrix()
        return feats

    @property
    def targets(self):
        return self.dataset[self.label]

    @property
    def targets_m(self):
        return self.targets.as_matrix()

    def _get_x_labels(self, num_x):
        return ['X%d' % (i + 1) for i in range(90)][num_x:]
