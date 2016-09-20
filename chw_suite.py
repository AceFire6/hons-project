import argparse
from datetime import datetime
import itertools
import json

from chw_data import CHWData
from util import generate_n_rgb_colours, get_short_codes, print_title, round_up

from joblib import Parallel, delayed
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy
from numpy import ndarray
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def param_run(debug_label=None, num_x=90, cross_folds=10, col_select=None,
              drop_cols=list(), test_data=None):
    results = []
    debug_label = '[{}] '.format(debug_label if debug_label else '')
    for estimator_name, estimator in estimators.iteritems():
        print '%sStarting %s' % (debug_label, estimator_name)
        feature_data = chw_data.get_features(num_x, col_select).drop(drop_cols)
        target_data = chw_data.get_targets(col_select)
        score = cross_validate_score(estimator, feature_data, target_data,
                                     test_data=test_data, cv=cross_folds)
        print '     %s Accuracy: %0.2f (+/- %0.3f)' % (estimator_name,
                                                       score.mean(),
                                                       score.std())
        results.append((estimator_name, score))
    return results


def draw_graph(graph_scores, x_values, y_lim=(0, 1), x_lim=None, y_label='',
               x_label='', file_name='', grid=True):
    min_err = 0.02
    legend = []
    x_lim = x_lim or (0, round_up(len(x_values)))

    x_range = x_values
    if type(x_values[0]) not in [int, float]:
        x_range = range(1, len(x_values) + 1)

    plt.ioff()
    plt.grid(grid)
    for graph_label, graph_vals in graph_scores.iteritems():
        y_vals, y_err = graph_vals
        y_err = y_err if any(map(lambda x: x > min_err, y_err)) else None
        plt.errorbar(x_range, y_vals, yerr=y_err, fmt=next(styles),
                     c=next(colours), linewidth=1.5, markersize=7,
                     markeredgewidth=1)
        legend.append(graph_label.replace('_', ' '))

    plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1))
    if type(x_values[0]) in [int, float]:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=1.0))
    else:
        plt.xticks(x_range, x_values)
    plt.legend(legend, loc=4)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name)
    plt.clf()


def draw_graph_from_file(experiment, split=False):
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

    if split:
        for key, val in file_scores.iteritems():
            conf = {k: v if k != 'file_name' else '%s-%s' % (key, v)
                    for k, v in config.iteritems()}
            draw_graph({key: val}, **conf)
    else:
        draw_graph(file_scores, **config)


def write_out_results(experiment, results, x_values, x_label, y_label,
                      file_name=None, draw=True):
    date = datetime.utcnow().replace(microsecond=0).isoformat()
    file_name = file_name or '%s-%s-graph.png' % (experiment, date)

    config = {
        'x_values': x_values, 'file_name': file_name, 'y_label': y_label,
        'x_label': x_label,
    }

    results_list = {n: map(ndarray.tolist, results[n]) for n in results}
    with open('%s-%s-config.json' % (experiment, date), 'w') as config_file:
        config_file.write(json.dumps(dict(config, results=results_list)))

    if draw:
        agg_scores = {i: [[], []] for i in results.keys()}
        for name, values in results.iteritems():
            for val in values:
                agg_scores[name][0].append(val.mean())
                agg_scores[name][1].append(val.std())
        draw_graph(agg_scores, **config)


def fit_and_score(estimator, feature_data, target_data, train_indices,
                  test_indices, test_data=None):
    features = feature_data.iloc[train_indices]
    targets = target_data.iloc[train_indices]
    estimator.fit(features, targets)

    test_features = (test_data.features
                     if test_data else feature_data).iloc[test_indices]
    test_targets = (test_data.targets
                    if test_data else target_data).iloc[test_indices]
    return estimator.score(test_features, test_targets)


def cross_validate_score(estimator, feature_data, target_data, test_data=None,
                         cv=10):
    kfold = StratifiedKFold(n_splits=cv)
    parallel = Parallel(n_jobs=-1)
    split = kfold.split(feature_data, target_data)
    scores = parallel(delayed(fit_and_score)(
        clone(estimator), feature_data, target_data, train, test, test_data)
             for train, test in split)
    return numpy.array(scores)


def effect_of_day_data_experiment():
    print_title('Running Effect of Day Experiment', '-')
    out_results = {i: [] for i in estimators.keys()}
    # Go through all values of X (1-90)
    x_val_range = range(1, 91)
    for x in x_val_range:
        result_scores = param_run(debug_label=x, num_x=x, cross_folds=10)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    write_out_results('xvals', out_results, x_val_range,
                      'Number of days included', 'Accuracy')


def region_generalization_experiment():
    print_title('Running Region Generalization Experiment', '-')
    region_data_size = 500
    countries = chw_data._dataset.groupby('country').size().sort_values()
    countries = countries[countries > region_data_size].keys().tolist()
    out_results = {name: [] for name in estimators.keys()}

    for country in countries:
        col_select = {country: 1}
        result_scores = param_run(debug_label=country, cross_folds=10,
                                  col_select=col_select, test_data=chw_data)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)
    countries = get_short_codes(countries)
    write_out_results('region', out_results, countries, 'Country', 'Accuracy')


def sector_generalization_experiment():
    print_title('Running Region Generalization Experiment', '-')
    sectors = chw_data._dataset.sector.unique()
    sectors = [sector for sector in sectors if type(sector) == str]
    out_results = {name: [] for name in estimators.keys()}

    for sector in sectors:
        col_select = {sector: 1}
        result_scores = param_run(debug_label=sector, cross_folds=10,
                                  col_select=col_select, test_data=chw_data)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    write_out_results('sector', out_results, sectors, 'Sector', 'Accuracy')


if __name__ == '__main__':
    experiments = [
        '0. Effect of Number of Days Included (1-90)',
        '1. Ability to Generalize Region Data',
        '2. Ability to Generalize Sector Data',
    ]
    experiment_functions = [
        effect_of_day_data_experiment,
        region_generalization_experiment,
        sector_generalization_experiment,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', dest='experiments', type=int,
                        nargs='*', choices=range(len(experiments)),
                        help='Choose which experiments to run as list',)
    parser.add_argument('-g', '--graph', dest='graph_file', type=str,
                        help='Graph values from experiment',)
    parser.add_argument('-s', '--split', action='store_true',
                        help='Generate a separate graph for each estimator')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all experiments')
    args = parser.parse_args()

    matplotlib.rcParams['backend'] = "Qt4Agg"
    markers = ['o', '^', 's', 'D']
    line_types = ['-']
    styles = [''.join(i) for i in itertools.product(line_types, markers)]
    styles = itertools.cycle(styles)

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
        'Decision_Tree': tree,
        'Random_Forest': forest,
        'SVM': svm,
        'Neural_Network': nn,
    }

    colours = iter(generate_n_rgb_colours(len(estimators)))

    if args.list:
        print_title('All Experiments:', '-')
        print '\n'.join(experiments)
    elif args.graph_file:
        print 'Drawing graph: %s' % args.graph_file
        draw_graph_from_file(args.graph_file, args.split)
    elif not args.experiments:
        print_title('Running All Experiments', '=')
        map(lambda func: func(), experiment_functions)
    elif args.experiments:
        for exp_no in range(len(experiment_functions)):
            if exp_no in args.experiments:
                experiment_functions[exp_no]()
