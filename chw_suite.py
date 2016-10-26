import argparse
from datetime import datetime
import itertools
import json
import sys

from experiment_data import ExperimentData
from util import (calculate_false_negatives, generate_n_rgb_colours,
                  replace_isodate, get_short_codes, get_split_and_balance,
                  list_index, print_title, round_up)

from imblearn.over_sampling import ADASYN
from joblib import Parallel, delayed
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def param_run(feature_data, target_data, test_features=None, test_targets=None,
              debug_label=None, cross_folds=10, repeat_test=False,
              balance=True):
    too_few_classes = len(target_data.unique()) == 1
    if test_targets is not None:
        too_few_classes = too_few_classes or len(test_targets.unique()) == 1

    if too_few_classes:
        if args.test:
            print 'Too few classes, not running'
            return None

    results = {'values': []}
    test_type = 'Repetitions' if repeat_test else 'Cross Evaluation'
    debug_str = '[{}] '.format(debug_label) if debug_label else ''
    info_vals = get_split_and_balance(target_data, test_targets)
    info_str = ('{}[N-train: {train_n} | N-test: {test_n} | Balance(+/-): '
                'Train {pos_train}/{neg_train} - Test {pos_test}/{neg_test}]')
    print info_str.format(debug_str, **info_vals)
    info_vals['label'] = debug_label
    results['stats'] = info_vals

    if balance and repeat_test:
        adasyn = ADASYN()
        feature_data, target_data = adasyn.fit_sample(feature_data,
                                                      target_data)
        test_features, test_targets = adasyn.fit_sample(test_features,
                                                        test_targets)
        bal_str = ('\tBalanced to: (True/False) '
                   'Train: {pos_train}/{neg_train} '
                   'Test: {pos_test}/{neg_test}')
        val_splits = get_split_and_balance(target_data, test_targets)
        results['balanced_stats'] = val_splits
        print (bal_str.format(**val_splits))

    for estimator_name, estimator in estimators.iteritems():
        print '\tStarting %s - %s' % (estimator_name, test_type)
        if not repeat_test:
            score = cross_validate_score(estimator, feature_data,
                                         target_data, cv=cross_folds)
        else:
            score = repeat_validate_score(estimator, feature_data, target_data,
                                          test_features, test_targets)
        accuracy_report = (estimator_name, score['accuracy'].mean(),
                           score['accuracy'].std())
        fn_report = (estimator_name, score['false_negatives'].mean(),
                     score['false_negatives'].std())
        print '\t\t%s Accuracy: %0.2f (+/- %0.3f)' % accuracy_report
        print '\t\t%s False Negatives: %0.2f (+/- %0.3f)' % fn_report
        results['values'].append((estimator_name, score))
    return results


def draw_graph(graph_scores, x_values, y_lim=(0, 1), x_lim=None, y_label='',
               x_label='', x_tick_indices=None, file_name='', grid=True,
               max_x_len=5):
    file_name = replace_isodate(file_name, args.file_repl)
    legend = []
    legend_loc = 4

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

        legend_loc = legend_loc and (4 if all(i > 0.2 for i in y_vals) else 0)
        legend.append(graph_label.replace('_', ' '))

    plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1))
    if x_tick_indices or type(x_values[0]) not in [int, float]:
        labels = list_index(x_tick_indices, x_values)
        shorten_x_labels = any(len(str(i)) > max_x_len for i in labels)
        if shorten_x_labels:
            with open(file_name.replace('png', 'txt'), 'w') as label_f:
                x_map = [(i + 1, val) for i, val in enumerate(labels)]
                label_f.write(json.dumps(x_map))
            labels = [i[0] for i in x_map]
        plt.xticks(x_range, labels)
    else:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=1.0))

    plt.legend(legend, loc=legend_loc, fontsize='small', labelspacing=0.2)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name)
    plt.clf()


def draw_graph_from_file(experiment, split=False, x_ticks=list()):
    with open('%s-config.json' % experiment) as config_file:
        config = json.loads(config_file.readline())

    results = config.pop('results', {})

    x_len = len(config['x_values'])
    for x in x_ticks:
        if x < 0 or x > x_len:
            sys.exit('x_tick index: %d out of x_value range 0-%d' % (x, x_len))
    config['x_tick_indices'] = x_ticks
    print 'Drawing graph for %s' % experiment

    file_scores = {metric: {} for metric in results.keys()}
    for metric, result in results.iteritems():
        for classifier, scores in result.iteritems():
            file_scores[metric][classifier] = [[], []]
            for score in scores:
                score = numpy.array(score)
                file_scores[metric][classifier][0].append(score.mean())
                file_scores[metric][classifier][1].append(score.std())

    directory = '{}/{}/'.format(*args.graph.split('/')) if args.graph else ''
    file_name = config['file_name']
    for metric, result in file_scores.iteritems():
        if metric != 'accuracy' and not args.add_metrics:
            continue
        config['y_label'] = metric.replace('_', ' ').title()
        config['file_name'] = directory + metric + '_' + file_name
        draw_graph(result, **config)
        if split:
            f_n = 'false_negatives'
            for key, val in result.iteritems():
                values = {key: val}
                if metric != f_n and f_n in file_scores.keys():
                    values['%s_%s' % (key, f_n)] = file_scores[f_n][key]
                config['file_name'] = (directory + metric + '_' +
                                       key + '-' + file_name)
                draw_graph(values, **config)


def write_out_results(experiment, results, x_values, x_label, y_label,
                      file_name=None, draw=False):
    if not args.file_repl:
        date = datetime.utcnow().replace(microsecond=0).isoformat()
    else:
        date = args.file_repl
    file_name = file_name or '%s-%s-graph.png' % (experiment, date)

    config = {
        'x_values': x_values, 'file_name': file_name, 'y_label': y_label,
        'x_label': x_label,
    }

    results_list = {}
    with open('%s-%s-info.txt' % (experiment, date), 'w') as info_file:
        for result in results:
            if not result:
                continue
            expr_label = '%s - ' % result['stats'].pop('label')
            info_file.write(expr_label + json.dumps(result['stats']) + '\n')
            if 'balanced_stats' in result:
                info_file.write(
                    expr_label + json.dumps(result['balanced_stats']) + '\n'
                )
            for estimator, metrics in result['values']:
                for metric, values in metrics.iteritems():
                    if metric not in results_list:
                        results_list[metric] = {}
                    if estimator in results_list[metric]:
                        results_list[metric][estimator].append(values.tolist())
                    else:
                        results_list[metric][estimator] = [values.tolist()]

    with open('%s-%s-config.json' % (experiment, date), 'w') as config_file:
        config_file.write(json.dumps(dict(config, results=results_list)))

    if draw:
        for metric, metric_results in results_list.iteritems():
            if metric != 'accuracy' and not args.add_metrics:
                continue
            agg_metric = {i: [[], []] for i in metric_results.keys()}
            for estimator, est_results in metric_results.iteritems():
                for values in est_results:
                    values = numpy.array(values)
                    agg_metric[estimator][0].append(values.mean())
                    agg_metric[estimator][1].append(values.std())
            file_name = '%s_%s' % (metric, config['file_name'])
            y_label = metric.replace('_', ' ').title()
            metric_config = dict(config, y_label=y_label, file_name=file_name)
            draw_graph(agg_metric, **metric_config)
    return dict(config, results=results_list)


def draw_table(data, projects, html=False):
    headings = ('Project Code', 'Model Constraint', 'Relative Accuracy',
                'Accuracy')
    head_template = (
        '<tr class="{_class}">%s</tr>\n' % ('<th>%s</th>' * len(headings))
        if html else ('| %s ' * len(headings)) + '|\n'
    )
    row_template = head_template.replace('th', 'td') if html else head_template

    model_constraints = ['All Data', 'Same Country',
                         'Same Sector', 'Same Country & Sector']

    for name, values in data.iteritems():
        print 'Table %s' % name
        if html:
            table = '<thead>%s</thead>\n<tbody>\n' % (head_template % headings)
        else:
            table = (head_template % headings).format(_class='')
        constraint_index = 0
        project_index = 0
        base_accuracy = 0
        for value in values:
            project = projects[project_index]
            label = model_constraints[constraint_index]
            accuracy = (sum(value) / len(value)) * 100 if value else 0
            relative_accuracy = 0
            if constraint_index == 0:
                base_accuracy = accuracy
            else:
                relative_accuracy = accuracy - base_accuracy

            _class = (
                (relative_accuracy and
                 ('pos' if relative_accuracy > 0 else 'neg')) or 'neutral'
            )

            relative_accuracy = '{:+.3f}'.format(relative_accuracy)
            accuracy = '{:.3f}'.format(accuracy)
            this_row = row_template % (project, label,
                                       relative_accuracy, accuracy)

            table += this_row.format(_class=_class)
            constraint_index = (constraint_index + 1) % 4
            if constraint_index == 0:
                project_index += 1
        html_temp = '<table class="pure-table pure-table-bordered">%s</table>'
        print html_temp % (table + '</tbody>') if html else table


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
    classification = estimator.predict(test_features)
    false_negatives = calculate_false_negatives(classification, test_targets)
    return estimator.score(test_features, test_targets), false_negatives


def repeat_validate_score(estimator, feature_data, target_data, test_features,
                          test_targets, repetitions=5):
    parallel = Parallel(n_jobs=-1)
    results = parallel(
        delayed(fit_and_score)(
            clone(estimator), feature_data, target_data,
            test_features=test_features, test_targets=test_targets
        ) for i in range(repetitions)
    )
    scores = []
    false_negatives = []
    for score, false_negative in results:
        scores.append(score)
        false_negatives.append(false_negative)
    return {'accuracy': numpy.array(scores),
            'false_negatives': numpy.array(false_negatives)}


def cross_validate_score(estimator, feature_data, target_data, cv=10):
    kfold = StratifiedKFold(n_splits=cv)
    parallel = Parallel(n_jobs=-1)
    split = kfold.split(feature_data, target_data)
    results = parallel(
        delayed(fit_and_score)(
            clone(estimator), feature_data, target_data, train, test
        ) for train, test in split
    )
    scores = []
    false_negatives = []
    for score, false_negative in results:
        scores.append(score)
        false_negatives.append(false_negative)
    return {'accuracy': numpy.array(scores),
            'false_negatives': numpy.array(false_negatives)}


def all_data_performance_experiment():
    """All Data Performance Experiment"""
    print_title('All Data Performance Experiment', '-')
    feature_data = chw_data.get_features()
    target_data = chw_data.get_targets()
    result_scores = param_run(feature_data, target_data)
    return result_scores


def added_features_performance_experiment():
    """Added features Performance Experiment"""
    print_title('Added Features Performance Experiment', '-')

    value_columns = ['X%d' % i for i in range(1, 91)]
    total_forms = chw_data.dataset[value_columns].sum(axis=1)
    pct_utilized = (chw_data.dataset[value_columns] > 0).sum(axis=1) / 90
    active_last_month = (chw_data.dataset[value_columns[-31:]] > 0).any(axis=1)
    prev_10_gradient = (chw_data.dataset['X90'] - chw_data.dataset['X80']) / 10

    # All the data without any new features
    out_results = [all_data_performance_experiment()]

    other_features = {
        'total_forms': total_forms,
        'activity_in_last_month': active_last_month,
        'pct_days_utilized': pct_utilized,
        'gradient_last_10_days': prev_10_gradient
    }

    for feature_name, feature_column in other_features.iteritems():
        feature_column.rename(feature_name, inplace=True)
        chw_data.add_feature(feature_name, feature_column)
        result_scores = param_run(chw_data.features, chw_data.targets,
                                  debug_label=feature_name)
        out_results.append(result_scores)
        chw_data.drop_feature(feature_name)

    x_values = ['None'] + other_features.keys()
    write_out_results('features', out_results, x_values, 'Feature', 'Accuracy',
                      draw=args.graph)


def effect_of_day_data_experiment():
    """Effect of Number of Days Included (1-90)"""
    print_title('Running Effect of Day Experiment', '-')
    out_results = []
    # Go through all values of X (1-90)
    x_val_range = range(1, 91)
    for x in x_val_range:
        x_col = ['nX%d' % i for i in range(x + 1, 91)]
        feature_data = chw_data.get_features(drop_cols=chw_data.categories)
        feature_data = feature_data.drop(x_col, axis=1)
        target_data = chw_data.get_targets()
        result_scores = param_run(feature_data, target_data,
                                  debug_label=x, cross_folds=10)
        out_results.append(result_scores)
    write_out_results('days', out_results, x_val_range,
                      'Number of days included', 'Accuracy', draw=args.graph)


def country_to_all_generalization_experiment(inverse=False):
    """Ability to Generalize Country Data"""
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Country Generalization Experiment', '-', inverse)
    region_data_size = 500
    countries = [key for key, val in chw_data.country.iteritems()
                 if val > region_data_size]
    out_results = []

    for country in countries:
        col_select = {country: 1}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_select, exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if not inverse:
            result_scores = param_run(feature_data, target_data,
                                      test_features=test_features,
                                      test_targets=test_targets,
                                      debug_label=country, repeat_test=True)
        else:
            result_scores = param_run(test_features, test_targets,
                                      test_features=feature_data,
                                      test_targets=target_data,
                                      debug_label=country, repeat_test=True)
        out_results.append(result_scores)
    countries = get_short_codes(countries)
    experiment = 'country' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, countries, 'Country',
                      'Accuracy', draw=args.graph)


def all_to_country_generalization_experiment():
    """Ability to Generalize Data to Country"""
    country_to_all_generalization_experiment(True)


def sector_to_all_generalization_experiment(inverse=False):
    """Ability to Generalize Sector Data"""
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Country Generalization Experiment', '-', inverse)
    sectors = set(chw_data.sector)
    sectors = [sector for sector in sectors if type(sector) == str]
    out_results = []

    for sector in sectors:
        col_select = {sector: 1}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_filter=col_select,
                                              exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if not inverse:
            result_scores = param_run(feature_data, target_data,
                                      test_features=test_features,
                                      test_targets=test_targets,
                                      debug_label=sector, repeat_test=True)
        else:
            result_scores = param_run(test_features, test_targets,
                                      test_features=feature_data,
                                      test_targets=target_data,
                                      debug_label=sector, repeat_test=True)
        out_results.append(result_scores)

    experiment = 'sector' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, sectors, 'Sector', 'Accuracy',
                      draw=args.graph)


def all_to_sector_generalization_experiment():
    """Ability to Generalize Data to Sector"""
    sector_to_all_generalization_experiment(True)


def project_to_all_generalization_experiment(inverse=False):
    """Ability to Generalize Project Data"""
    inverse = (inverse and 'Inverse ') or ''
    print_title('Running {}Project Generalization Experiment', '-', inverse)
    project_codes = chw_data.get_column_values('projectCode', top_n=10)
    out_results = []

    for project_code in project_codes:
        col_select = {'projectCode': project_code}
        feature_data = chw_data.get_features(col_select)
        target_data = chw_data.get_targets(col_select)
        test_features = chw_data.get_features(col_filter=col_select,
                                              exclude=True)
        test_targets = chw_data.get_targets(col_select, exclude=True)
        if not inverse:
            result_scores = param_run(feature_data, target_data,
                                      test_features=test_features,
                                      test_targets=test_targets,
                                      debug_label=project_code,
                                      repeat_test=True)
        else:
            result_scores = param_run(test_features, test_targets,
                                      test_features=feature_data,
                                      test_targets=target_data,
                                      debug_label=project_code,
                                      repeat_test=True)

        out_results.append(result_scores)

    experiment = 'project' + ('_inverse' if inverse else '')
    write_out_results(experiment, out_results, map(str, project_codes.keys()),
                      'Project', 'Accuracy', draw=args.graph)


def all_to_project_generalization_experiment():
    """Ability to Generalize Data to Project"""
    project_to_all_generalization_experiment(True)


def project_model_comparison_experiment():
    """Project Model Comparison Table"""
    print_title('Running Project Model Comparison Experiment', '-')
    training_order = ['All Data', 'Same Country',
                      'Same Sector', 'Same Country & Sector']
    out_results = []

    project_codes = chw_data.get_column_values('projectCode', top_n=10)

    for project_code in project_codes:
        col_select = {'projectCode': project_code}
        all_f = chw_data.get_features(col_select, exclude=True)
        all_t = chw_data.get_targets(col_select, exclude=True)
        all_data = (all_f, all_t)
        country, sector = chw_data.get_columns(
            lambda x: all(x == 1),
            col_select
        )

        same_country = (all_f[all_f[country] == 1], all_t[all_f[country] == 1])
        same_sector = (all_f[all_f[sector] == 1], all_t[all_f[sector] == 1])
        same_country_and_sector = (
            all_f[(all_f[country] == 1) & (all_f[sector] == 1)],
            all_t[(all_f[country] == 1) & (all_f[sector] == 1)],
        )
        test_features = chw_data.get_features(col_select)
        test_targets = chw_data.get_targets(col_select)
        training_sets = {'All Data': all_data, 'Same Country': same_country,
                         'Same Sector': same_sector,
                         'Same Country & Sector': same_country_and_sector}
        for name in training_order:
            label = '%d - %s' % (project_code, name)
            train_features, train_targets = training_sets[name]
            if len(train_targets.value_counts()) > 1:
                result_scores = param_run(train_features, train_targets,
                                          test_features=test_features,
                                          test_targets=test_targets,
                                          debug_label=label, repeat_test=True)
            else:
                result_scores = {'stats': {'label': label}, 'values': ()}
                print 'Too few classes'
            out_results.append(result_scores)
    write_out_results('combo', out_results, project_codes, None, 'Accuracy')


def clean_dataset(dataset):
    # Drop duplicate users
    dataset.drop_duplicates('userCode', inplace=True)
    # Replace blank sector fields with No info
    dataset.sector.fillna('No info', inplace=True)
    # Replace NaN in country column with No info
    dataset.country.fillna('Not Specified', inplace=True)
    # If in tests mode only let a fraction of data through
    return dataset.sample(frac=args.test) if args.test else dataset


if __name__ == '__main__':
    experiment_functions = [
        all_data_performance_experiment,
        added_features_performance_experiment,
        effect_of_day_data_experiment,
        country_to_all_generalization_experiment,
        all_to_country_generalization_experiment,
        sector_to_all_generalization_experiment,
        all_to_sector_generalization_experiment,
        project_to_all_generalization_experiment,
        all_to_project_generalization_experiment,
        project_model_comparison_experiment,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', dest='experiments', type=int,
                        nargs='*', choices=range(len(experiment_functions)),
                        help='Choose which experiments to run as list',)
    parser.add_argument('-g', '--graph', dest='graph', nargs='?',
                        default=False, help='Graph values from experiment')
    parser.add_argument('-f', '--file', dest='file_repl',
                        help='Replace date in generated file names')
    parser.add_argument('-x', dest='x_ticks', type=int, nargs='*', default=[],
                        help='Select which x axis features to show by index',)
    parser.add_argument('-s', '--split', action='store_true',
                        help='Generate a separate graph for each estimator')
    parser.add_argument('-t', '--test', nargs='?', default=False, type=float,
                        help=('Run in test mode. '
                              'Uses specified fraction of data or 0.25'))
    parser.add_argument('-m', '--more-metrics', action='store_true',
                        dest='add_metrics', help='Graph additional metrics.')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all experiments')
    args = parser.parse_args()

    # Handle argument with optional value
    if not args.test:
        args.test = (args.test is None) and 0.25  # 0.25 if no explicit value

    matplotlib.rcParams['backend'] = "Qt4Agg"
    markers = ['o', '^', 's', 'D', 'v']
    line_types = ['-']
    styles = [''.join(i) for i in itertools.product(line_types, markers)]
    styles = itertools.cycle(styles)

    label = 'activeQ2'
    drop_features = ['projectCode', 'userCode']
    categorical_features = ['country', 'sector']

    chw_data = ExperimentData('chw_data.csv', label, drop_features,
                              categorical_features, clean_dataset)

    tree = DecisionTreeClassifier(class_weight='balanced')
    forest = RandomForestClassifier(class_weight='balanced')
    svm = SVC(class_weight='balanced', cache_size=1000)
    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    ridge = RidgeClassifier(class_weight='balanced')

    estimators = {
        'Decision_Tree': tree,
        'Random_Forest': forest,
        'SVM': svm,
        'Neural_Network': nn,
        'Ridge': ridge,
    }

    colours = itertools.cycle(generate_n_rgb_colours(len(estimators)))

    if args.list:
        print_title('All Experiments:', '-')
        for i in range(len(experiment_functions)):
            print '%d. %s' % (i, experiment_functions[i].__doc__)
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
