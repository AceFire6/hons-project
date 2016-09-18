import argparse
from datetime import datetime
import itertools
import json

from chw_data import CHWData
from util import generate_n_rgb_colours, round_up

import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
        print '     %s Accuracy: %0.2f (+/- %0.3f)' % (estimator_name,
                                                       cross_score.mean(),
                                                       cross_score.std())
        results.append((estimator_name, cross_score))
    return results


def draw_graph(graph_scores, x_values, y_lim=(0, 1), x_lim=None,
               y_label='', x_label='', file_name='', min_err=0.2, grid=True):
    legend = []
    colours = iter(generate_n_rgb_colours(len(estimators)))
    markers = ['o', '^', 's', 'D']
    line_types = ['-', '--', ':']
    styles = [''.join(i) for i in itertools.product(line_types, markers)]
    styles = itertools.cycle(styles)
    x_lim = x_lim or (0, round_up(len(x_values)))

    plt.ioff()
    plt.grid(grid)
    for graph_label, graph_vals in graph_scores.iteritems():
        y_vals, y_err = graph_vals
        y_err = y_err if any(map(lambda x: x > min_err, y_err)) else None
        plt.errorbar(x_values, y_vals, yerr=y_err, fmt=next(styles),
                     c=next(colours), linewidth=1.5, markersize=7,
                     markeredgewidth=1)
        legend.append(graph_label.replace('_', ' '))

    plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=1.0))
    plt.legend(legend, loc=4)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name)


def draw_graph_from_file(experiment):
    file_scores = {}
    with open('%s-config.json' % experiment) as config_file:
        config = json.loads(config_file.readline())

    results = config.pop('results', [])

    for classifier, scores in results.iteritems():
        file_scores[classifier] = [[], []]
        for score in scores:
            score = numpy.array(score)
            file_scores[classifier][0].append(score.mean())
            file_scores[classifier][1].append(score.std())

    draw_graph(file_scores, **config)


def write_out_results(experiment, results, x_values, x_label, y_label, date,
                      file_name=None, draw=True):
    file_name = file_name or '%s-%s-graph.png' % (experiment, date)
    agg_scores = {i: [[], []] for i in results.keys()}

    for name, values in results.iteritems():
        for val in values:
            agg_scores[name][0].append(val.mean())
            agg_scores[name][1].append(val.std())

    config = {
        'x_values': x_values, 'file_name': file_name, 'y_label': y_label,
        'x_label': x_label,
    }

    if draw:
        draw_graph(agg_scores, **config)

    config['results'] = {n: map(ndarray.tolist, results[n]) for n in results}
    with open('%s-%s-config.json' % (experiment, date), 'w') as config_file:
        config_file.write(json.dumps(config))


def effect_of_day_data_experiment():
    print('Running Effect of Day Experiment\n--------------------------------')
    date = datetime.utcnow().replace(microsecond=0).isoformat()
    out_results = {i: [] for i in estimators.keys()}
    # Go through all values of X (1-90)
    x_val_range = range(1, 91)
    for x in x_val_range:
        result_scores = param_run(num_x=x, cross_folds=10)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    write_out_results('xvals', out_results, x_val_range,
                      'Number of days included', 'Accuracy', date)


if __name__ == '__main__':
    experiments = [
        '0. Effect of Number of Days Included (1-90)',
    ]
    experiment_functions = [
        effect_of_day_data_experiment,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', dest='experiments', type=int,
                        nargs='*', choices=range(len(experiments)),
                        help='Choose which experiments to run as list',)
    parser.add_argument('-g', '--graph', dest='graph_file', type=str,
                        help='Graph values from experiment',)
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all experiments')
    args = parser.parse_args()

    matplotlib.rcParams['backend'] = "Qt4Agg"
    label = 'activeQ2'
    drop_features = ['projectCode', 'userCode']
    categorical_features = ['country', 'sector']

    chw_data = CHWData('chw_data.csv', label, drop_features,
                       categorical_features)

    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier()
    svm = SVC()
    nn = MLPClassifier(hidden_layer_sizes=(50, 50))

    estimators = {
        'Decision_Tree': tree, 'Random_Forest': forest,
        'SVM': svm, 'Neural_Network': nn,
    }

    if args.list:
        print 'All Experiments:\n----------------'
        print '\n'.join(experiments)
    elif args.graph_file:
        print 'Drawing graph: %s' % args.graph_file
        draw_graph_from_file(args.graph_file)
    elif not args.experiments:
        print 'Running All Experiments\n=======================\n'
        map(lambda func: func(), experiment_functions)
    elif args.experiments:
        for exp_no in range(len(experiment_functions)):
            if exp_no in args.experiments:
                experiment_functions[exp_no]()
