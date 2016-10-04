import argparse
from datetime import datetime
import itertools
import json
import sys

from experiment_data import ExperimentData
from util import (generate_n_rgb_colours, get_short_codes, list_index,
                  print_title, round_up)

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


def param_run(feature_data, target_data, test_features=None, test_targets=None,
              debug_label=None, cross_folds=10, repeat_test=False):
    results = []
    debug_label = '[{}] '.format(debug_label) if debug_label else ''
    for estimator_name, estimator in estimators.iteritems():
        test_type = 'Repetitions' if repeat_test else 'Cross Evaluation'
        print '%sStarting %s - %s' % (debug_label, estimator_name, test_type)
        if not repeat_test:
            score = cross_validate_score(estimator, feature_data,
                                         target_data, cv=cross_folds)
        else:
            score = repeat_validate_score(estimator, feature_data, target_data,
                                          test_features, test_targets)
        accuracy_report = (' ' * 4, estimator_name, score.mean(), score.std())
        print '%s%s Accuracy: %0.2f (+/- %0.3f)' % accuracy_report
        results.append((estimator_name, score))
    return results


def draw_graph(graph_scores, x_values, y_lim=(0, 1), x_lim=None, y_label='',
               x_label='', x_tick_indices=None, file_name='', grid=True):
    legend = []

    x_range = x_values
    if type(x_values[0]) not in [int, float]:
        x_range = range(1, len(x_values) + 1)
        x_lim = x_lim or (0, round_up(len(x_range)))

    x_range = list_index(x_tick_indices, x_range)
    x_lim = x_lim or (0, round_up(x_range[-1]))

    plt.ioff()
    plt.grid(grid)
    for graph_label, graph_vals in graph_scores.iteritems():
        y_vals, y_err = list_index(x_tick_indices, *graph_vals)
        plt.errorbar(x_range, y_vals, yerr=y_err, fmt=next(styles),
                     c=next(colours), linewidth=1.5, markersize=7,
                     markeredgewidth=1)
        legend.append(graph_label.replace('_', ' '))

    plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1))
    if x_tick_indices or type(x_values[0]) not in [int, float]:
        labels = list_index(x_tick_indices, x_values)
        plt.xticks(x_range, labels)
    else:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=1.0))

    plt.legend(legend, loc=4)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name)
    plt.clf()


def draw_graph_from_file(experiment, split=False, x_ticks=list()):
    file_scores = {}
    with open('%s-config.json' % experiment) as config_file:
        config = json.loads(config_file.readline())

    results = config.pop('results', [])

    x_len = len(config['x_values'])
    for x in x_ticks:
        if x < 0 or x > x_len:
            sys.exit('x_tick index: %d out of x_value range 0-%d' % (x, x_len))
    config['x_tick_indices'] = x_ticks
    print 'Drawing graph for %s' % experiment

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
                      file_name=None, draw=False):
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


def fit_and_score(estimator, feature_data, target_data, train_indices=None,
                  test_indices=None, test_features=None, test_targets=None):
    if (train_indices is not None) and (test_indices is not None):
        train_features = feature_data.iloc[train_indices]
        train_targets = target_data.iloc[train_indices]
        estimator.fit(train_features, train_targets)
        test_features = feature_data.iloc[test_indices]
        test_targets = target_data.iloc[test_indices]
    else:
        estimator.fit(feature_data, target_data)
    return estimator.score(test_features, test_targets)


def repeat_validate_score(estimator, feature_data, target_data, test_features,
                          test_targets, repetitions=5):
    parallel = Parallel(n_jobs=-1)
    scores = parallel(delayed(fit_and_score)(
        clone(estimator), feature_data, target_data,
        test_features=test_features, test_targets=test_targets)
                      for i in range(repetitions))
    return numpy.array(scores)


def cross_validate_score(estimator, feature_data, target_data, cv=10):
    kfold = StratifiedKFold(n_splits=cv)
    parallel = Parallel(n_jobs=-1)
    split = kfold.split(feature_data, target_data)
    scores = parallel(delayed(fit_and_score)(
        clone(estimator), feature_data, target_data, train, test)
             for train, test in split)
    return numpy.array(scores)


def effect_of_day_data_experiment():
    print_title('Running Effect of Day Experiment', '-')
    out_results = {i: [] for i in estimators.keys()}
    # Go through all values of X (1-90)
    x_val_range = range(1, 91)
    for x in x_val_range:
        feature_data = chw_data.get_features(x, drop_cols=chw_data.categories)
        target_data = chw_data.get_targets()
        result_scores = param_run(feature_data, target_data,
                                  debug_label=x, cross_folds=10)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    write_out_results('xvals', out_results, x_val_range,
                      'Number of days included', 'Accuracy', draw=args.graph)


def country_to_all_generalization_experiment(inverse=False):
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Region Generalization Experiment', '-', inverse)
    region_data_size = 500
    countries = [key for key, val in chw_data.country.iteritems()
                 if val > region_data_size]
    out_results = {name: [] for name in estimators.keys()}

    for country in countries:
        col_select = {country: 1}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_filter=col_select,
                                              exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if inverse:
            feature_data, target_data = test_features, test_targets
        result_scores = param_run(feature_data, target_data,
                                  test_features=test_features,
                                  test_targets=test_targets,
                                  debug_label=country, repeat_test=True)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)
    countries = get_short_codes(countries)
    experiment = 'country' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, countries, 'Country', 'Accuracy',
                      draw=args.graph)


def all_to_country_generalization_experiment():
    country_to_all_generalization_experiment(True)


def sector_to_all_generalization_experiment(inverse=False):
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Country Generalization Experiment', '-', inverse)
    sectors = set(chw_data.sector)
    sectors = [sector for sector in sectors if type(sector) == str]
    out_results = {name: [] for name in estimators.keys()}

    for sector in sectors:
        col_select = {sector: 1}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_filter=col_select,
                                              exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if inverse:
            feature_data, target_data = test_features, test_targets
        result_scores = param_run(feature_data, target_data,
                                  test_features=test_features,
                                  test_targets=test_targets,
                                  debug_label=sector, repeat_test=True)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    experiment = 'sector' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, sectors, 'Sector', 'Accuracy',
                      draw=args.graph)


def all_to_sector_generalization_experiment():
    sector_to_all_generalization_experiment(True)


def project_to_all_generalization_experiment(inverse=False):
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Project Generalization Experiment', '-', inverse)
    project_codes = chw_data.get_column_values('projectCode', top_n=10)
    out_results = {name: [] for name in estimators.keys()}

    for project_code in project_codes:
        col_select = {'projectCode': project_code}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_filter=col_select,
                                              exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if inverse:
            feature_data, target_data = test_features, test_targets
        result_scores = param_run(feature_data, target_data,
                                  test_features=test_features,
                                  test_targets=test_targets,
                                  debug_label=project_code, repeat_test=True)
        for result in result_scores:
            name, val = result
            out_results[name].append(val)

    experiment = 'project' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, map(str, project_codes.keys()),
                      'Project', 'Accuracy', draw=args.graph)


def all_to_project_generalization_experiment():
    project_to_all_generalization_experiment(True)


if __name__ == '__main__':
    experiments = [
        '0. Effect of Number of Days Included (1-90)',
        '1. Ability to Generalize Country Data',
        '2. Ability to Generalize Data to Country',
        '3. Ability to Generalize Sector Data',
        '4. Ability to Generalize Data to Sector',
        '5. Ability to Generalize Project Data',
        '6. Ability to Generalize Data to Project',
    ]
    experiment_functions = [
        effect_of_day_data_experiment,
        country_to_all_generalization_experiment,
        all_to_country_generalization_experiment,
        sector_to_all_generalization_experiment,
        all_to_sector_generalization_experiment,
        project_to_all_generalization_experiment,
        all_to_project_generalization_experiment,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', dest='experiments', type=int,
                        nargs='*', choices=range(len(experiments)),
                        help='Choose which experiments to run as list',)
    parser.add_argument('-g', '--graph', dest='graph', nargs='?',
                        default=False, help='Graph values from experiment')
    parser.add_argument('-x', dest='x_ticks', type=int, nargs='*',
                        help='Select which x axis features to show by index',)
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

    chw_data = ExperimentData('chw_data.csv', label, drop_features,
                              categorical_features)

    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier()
    svm = SVC()
    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)

    estimators = {
        'Decision_Tree': tree,
        'Random_Forest': forest,
        'SVM': svm,
        'Neural_Network': nn,
    }

    colours = itertools.cycle(generate_n_rgb_colours(len(estimators)))

    if args.list:
        print_title('All Experiments:', '-')
        print '\n'.join(experiments)
    elif args.graph:
        draw_graph_from_file(args.graph, args.split, args.x_ticks)
    elif args.experiments:
        args.graph = args.graph is None
        for exp_no in range(len(experiment_functions)):
            if exp_no in args.experiments:
                experiment_functions[exp_no]()
    elif args.experiments is not None:
        print_title('Running All Experiments', '=')
        map(lambda func: func(), experiment_functions)
    elif args.graph:
        print 'Nothing to graph. No file or experiment specified.'
    else:
        print 'No action specified. ' \
              'Run `python %s -l` to see all actions.' % __file__
