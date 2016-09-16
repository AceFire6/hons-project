import itertools
import json

from chw_data import CHWData
from util import generate_n_rgb_colours, round_up

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
drop_features = ['projectCode', 'userCode']
categorical_features = ['country', 'sector']

chw_data = CHWData('chw_data.csv', label, drop_features, categorical_features)

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


def param_run(num_x=90, cross_folds=10, drop_cols=list()):
    results = []
    for estimator_name, estimator in estimators.iteritems():
        print '[%d] Starting %s' % (num_x, estimator_name)
        feature_data = chw_data.get_features(num_x).drop(drop_cols)
        feature_data = (feature_data.as_matrix()
                        if estimator == nn else feature_data)
        cross_score = cross_val_score(estimator, n_jobs=-1,
                                      X=feature_data,
                                      y=chw_data.targets, cv=cross_folds)
        print '     %s Accuracy: %0.2f (+/- %0.2f)' % (estimator_name,
                                                       cross_score.mean(),
                                                       cross_score.std() * 2)
        results.append((estimator_name, cross_score))
    return results


def draw_graph(graph_scores, x_values, y_lim=(0, 1), x_lim=None,
               y_label='', x_label='', file_name='', min_err=0.2):
    legend = []
    colours = iter(generate_n_rgb_colours(len(estimators)))
    markers = ['o', '^', 's', 'D']
    line_types = ['-', '--', ':']
    styles = itertools.cycle(itertools.product(markers, line_types))

    x_lim = x_lim or (0, round_up(len(x_values)))

    plt.ioff()
    for graph_label, graph_vals in graph_scores.iteritems():
        y_vals, y_err = graph_vals
        y_err = y_err if any(map(lambda x: x > min_err, y_err)) else None
        plt.errorbar(x_values, y_vals, yerr=y_err, fmt=next(styles),
                     c=next(colours), linewidth=1.5, markersize=7,
                     markeredgewidth=1)
        legend.append(graph_label)

    plt.legend(legend, loc=4)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name)


def effect_of_day_data_experiment():
    all_scores = {i: [[], []] for i in estimators.keys()}
    # Go through all values of X (1-90)
    x_val_range = range(1, 91)
    for x in x_val_range:
        result_scores = param_run(num_x=x, cross_folds=10)
        for result in result_scores:
            name, val = result
            all_scores[name][0].append(val.mean())
            all_scores[name][1].append(val.std())
            with open('x-vals-%s-scores.json' % name, 'a+') as fout:
                fout.write(json.dumps(val.tolist()) + '\n')

    draw_graph(all_scores, x_val_range, file_name='x-vals-compare.png',
               y_label='Accuracy', x_label='Number of days included')


if __name__ == '__main__':
    print('Running Effect of Day Experiment')
    effect_of_day_data_experiment()
