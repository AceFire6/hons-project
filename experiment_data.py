import pandas
from pandas import DataFrame
from sklearn.preprocessing import normalize


class ExperimentData(object):
    def __init__(self, file_name, label='', drop_cols=list(),
                 categorical_features=list(), clean_func=None):
        self._file_name = file_name
        self._dataset = DataFrame.from_csv(file_name)
        self._features = self._dataset.columns
        self._processed_dataset = None
        self.drop_cols = drop_cols
        self.categories = categorical_features
        self.label = label
        self.clean_data = clean_func
        if label:
            if self.clean_data:
                self.clean_data(self._dataset)
            self._process_dataset(label, True, drop_cols, categorical_features)


    def _process_dataset(self, label, assign=True, drop_cols=list(),
                         categorical_features=list()):
        dataset = self._dataset
        features = dataset.columns.drop(drop_cols + [label])

        new_categorical_features = []
        for feature in categorical_features:
            dummies = pandas.get_dummies(dataset[feature])
            features = features.append(dummies.columns).drop(feature)
            dataset = pandas.concat([dataset, dummies], axis=1)
            feature_counts = dataset.groupby(feature).size().to_dict()
            setattr(self, feature, feature_counts)
            new_categorical_features.extend(dummies.columns.tolist())
            dataset.drop(feature, axis=1, inplace=True)

        if categorical_features:
            number_features = features.difference(new_categorical_features)
            self.categories = new_categorical_features
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

    def get_features(self, col_filter=None, as_matrix=False,
                     exclude=False, drop_cols=list()):
        features = self.features
        indices = self.get_indices(col_filter or {}, exclude)
        if not indices.empty:
            features = features[indices]
        features = features.drop(drop_cols, axis=1)
        return features.as_matrix() if as_matrix else features

    @property
    def targets(self):
        return self.dataset[self.label]

    @property
    def targets_m(self):
        return self.targets.as_matrix()

    def get_targets(self, col_filter=None, as_matrix=False, exclude=False):
        targets = self.targets
        indices = self.get_indices(col_filter or {}, exclude)
        if not indices.empty:
            targets = targets[indices]
        return targets.as_matrix() if as_matrix else targets

    def get_indices(self, col_select, exclude=False):
        results = pandas.Series()
        for key, val in col_select.iteritems():
            if exclude:
                results = results.append(self.dataset[key] != val)
            else:
                results = results.append(self.dataset[key] == val)
        return results

    def get_column_values(self, columns, top_n=None, occurance_min=None):
        grouped_columns = self.dataset.groupby(columns).size()
        if occurance_min:
            return grouped_columns[grouped_columns >= occurance_min].to_dict()
        elif top_n:
            sorted_groups = grouped_columns.sort_values(ascending=False)
            return sorted_groups.iloc[range(top_n)].to_dict()
        return grouped_columns.to_dict()

    def get_columns(self, func, col_filter=None):
        indices = self.get_indices(col_filter or {})
        data = self.dataset[indices] if (indices is not None) else self.dataset
        return [label for label, val in data.apply(func).iteritems() if val]
