import json
import sys
from datetime import datetime
import joblib
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy
from imblearn.over_sampling import ADASYN
from sklearn import clone
from sklearn.model_selection import StratifiedKFold

from util import (calculate_false_negatives_and_positives,
                  get_json_in_subfolders, get_split_and_balance, list_index,
                  replace_isodate, round_up)


def fit_and_score(estimator, feature_data, target_data,
                  train_indices=None, test_indices=None,
                  test_features=None, test_targets=None):
    if (train_indices is not None) and (test_indices is not None):
        train_features = feature_data.iloc[train_indices]
        train_targets = target_data.iloc[train_indices]
        estimator.fit(train_features, train_targets)
        test_features = feature_data.iloc[test_indices]
        test_targets = target_data.iloc[test_indices]
    else:
        estimator.fit(feature_data, target_data)
    classification = estimator.predict(test_features)
    false_counts = calculate_false_negatives_and_positives(classification,
                                                           test_targets)
    return (
        estimator, estimator.score(test_features, test_targets), false_counts
    )


def draw_table(data_file):
    with open(data_file) as data_in:
        data = json.loads(data_in.readline())

    projects = data.get('x_values', [])
    classifiers = data['results']['accuracy'].keys()

    headings = (
        'Project Code', 'Model Constraint', 'Relative', 'True',
    ) + (('Relative', 'True') * (len(classifiers) - 1))

    head_template = '<tr>%s</tr>\n' % ('<th>%s</th>' * len(headings))
    row_template = head_template.replace('th', 'td')

    model_constraints = (
        'All Data', 'Same Country', 'Same Sector', 'Same Country & Sector',
    )

    restruct = {}
    for metric, results in data['results'].iteritems():
        metric_struct = restruct[metric] = {}
        for classifier, values in results.iteritems():
            constraint_index = 0
            project_index = 0
            base_accuracy = 0
            for value in values:
                project = projects[project_index]
                constraint = model_constraints[constraint_index]

                if project not in metric_struct:
                    metric_struct[project] = {}

                if constraint not in metric_struct[project]:
                    metric_struct[project][constraint] = ()

                accuracy = (sum(value) / len(value)) * 100 if value else 0
                relative_accuracy = 0
                if constraint_index == 0:
                    base_accuracy = accuracy
                else:
                    relative_accuracy = accuracy - base_accuracy

                relative_accuracy = '{:+.3f}'.format(relative_accuracy)
                accuracy = '{:.3f}'.format(accuracy)

                # # To test ordering
                # metric_struct[project][constraint] += (
                #     (classifier, relative_accuracy, accuracy)
                # )

                metric_struct[project][constraint] += (
                    relative_accuracy, accuracy
                )

                constraint_index = (constraint_index + 1) % 4
                if constraint_index == 0:
                    project_index += 1

    for metric, project_data in restruct.iteritems():
        print 'Table for %s' % metric
        table_inner = '<thead>\n%s</thead>\n<tbody>\n' % (
            head_template % headings
        )

        for project, constraint_data in project_data.iteritems():
            for constraint, c_results in constraint_data.iteritems():
                table_inner += row_template % (
                    (project, constraint) + c_results
                )

        table_tag = (
            '<table class="pure-table pure-table-bordered">\n%s\n</table>'
        )
        print classifiers
        print table_tag % (table_inner + '</tbody>')


def repeat_validate_score(estimator, feature_data, target_data,
                          test_features, test_targets, repetitions=5):
    parallel = Parallel(n_jobs=-1)
    results = parallel(
        delayed(fit_and_score)(
            clone(estimator), feature_data, target_data,
            test_features=test_features, test_targets=test_targets
        ) for i in range(repetitions)
    )
    best_estimator = None
    best_score = -1
    scores = []
    false_negatives = []
    false_positives = []
    for estimator, score, false_counts in results:
        if score > best_score:
            best_estimator = estimator
            best_score = score
        scores.append(score)
        false_negatives.append(false_counts['negatives'])
        false_positives.append(false_counts['positives'])
    return {
        'accuracy': numpy.array(scores),
        'false_negatives': numpy.array(false_negatives),
        'false_positives': numpy.array(false_positives),
        'misc': {
            'best_estimator': {
                'estimator': best_estimator,
                'score': best_score,
            }
        }
    }


def cross_validate_score(estimator, feature_data, target_data,
                         cv=10):
    kfold = StratifiedKFold(n_splits=cv)
    parallel = Parallel(n_jobs=-1)
    split = kfold.split(feature_data, target_data)
    results = parallel(
        delayed(fit_and_score)(
            clone(estimator), feature_data, target_data, train, test
        ) for train, test in split
    )
    best_estimator = None
    best_score = -1
    scores = []
    false_negatives = []
    false_positives = []
    for estimator, score, false_counts in results:
        if score > best_score:
            best_estimator = estimator
            best_score = score
        scores.append(score)
        false_negatives.append(false_counts['negatives'])
        false_positives.append(false_counts['positives'])
    return {
        'accuracy': numpy.array(scores),
        'false_negatives': numpy.array(false_negatives),
        'false_positives': numpy.array(false_positives),
        'misc': {
            'best_estimator': {
                'estimator': best_estimator,
                'score': best_score,
            }
        }
    }


class ExperimentHelper(object):
    def __init__(self, estimators, args, colours, styles):
        self.estimators = estimators
        for key, val in vars(args).iteritems():
            setattr(self, key, val)
        self.colours = colours
        self.styles = styles

    def param_run(self, feature_data, target_data, test_features=None,
                  test_targets=None, debug_label=None, cross_folds=10,
                  repeat_test=False, balance=True):
        too_few_classes = len(target_data.unique()) == 1
        if test_targets is not None:
            too_few_classes = (
                too_few_classes or len(test_targets.unique()) == 1
            )

        if too_few_classes:
            if self.test:
                print 'Too few classes, not running'
                return None

        results = {'values': []}
        test_type = 'Repetitions' if repeat_test else 'Cross Evaluation'
        debug_str = '[{}] '.format(debug_label) if debug_label else ''
        info_vals = get_split_and_balance(target_data, test_targets)
        info_str = (
            '{}[N-train: {train_n} | N-test: {test_n} | Balance(+/-): '
            'Train {pos_train}/{neg_train} - Test {pos_test}/{neg_test}]'
        )
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

        best_estimator = {'obj': None, 'score': -1, 'name': ''}
        for estimator_name, estimator in self.estimators.iteritems():
            print '\tStarting %s - %s' % (estimator_name, test_type)
            if not repeat_test:
                score = cross_validate_score(
                    estimator, feature_data, target_data, cv=cross_folds
                )
            else:
                score = repeat_validate_score(
                    estimator, feature_data, target_data,
                    test_features, test_targets,
                )
            misc = score.pop('misc')
            if best_estimator['score'] < misc['best_estimator']['score']:
                best_estimator['obj'] = misc['best_estimator']['estimator']
                best_estimator['score'] = misc['best_estimator']['score']
                best_estimator['name'] = estimator_name

            accuracy_report = (estimator_name, score['accuracy'].mean(),
                               score['accuracy'].std())
            fn_report = (estimator_name, score['false_negatives'].mean(),
                         score['false_negatives'].std())
            fp_report = (estimator_name, score['false_positives'].mean(),
                         score['false_positives'].std())
            print '\t\t%s Accuracy: %0.2f (+/- %0.3f)' % accuracy_report
            print '\t\t%s False Negatives: %0.2f (+/- %0.3f)' % fn_report
            print '\t\t%s False Positives: %0.2f (+/- %0.3f)' % fp_report
            results['values'].append((estimator_name, score))
        print '\tBest estimator: %s' % best_estimator['name']
        return results, best_estimator

    def draw_graph(self, graph_scores, x_values, y_lim=(0, 1), x_lim=None,
                   y_label='', x_label='', x_tick_indices=None,
                   file_name='', grid=True, max_x_len=5):
        file_name = replace_isodate(file_name, self.file_repl)
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
            plt.errorbar(x_range, y_vals, yerr=y_err, fmt=next(self.styles),
                         c=next(self.colours), linewidth=1.5, markersize=7,
                         markeredgewidth=1)

            legend_loc = (
                legend_loc and (4 if all(i > 0.2 for i in y_vals) else 0)
            )
            legend.append(graph_label.replace('_', ' '))

        plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1))
        if x_tick_indices or type(x_values[0]) not in [int, float]:
            labels = list_index(x_tick_indices, x_values)
            shorten_x_labels = (
                any(len(str(i)) > max_x_len for i in labels) and
                len(labels) >= 5
            )
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
        plt.tight_layout()
        plt.savefig(file_name)
        plt.clf()

    def draw_graph_from_file(self, experiment, split=False, x_ticks=list()):
        with open('%s-config.json' % experiment) as config_file:
            config = json.loads(config_file.readline())

        results = config.pop('results', {})

        x_len = len(config['x_values'])
        for x in x_ticks:
            if x < 0 or x > x_len:
                sys.exit(
                    'x_tick index: %d out of x_value range 0-%d' % (x, x_len)
                )
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

        path_split = experiment.split('/') if experiment else ''
        directory = '{}/{}/'.format(*path_split) if len(path_split) > 1 else ''
        file_name = config['file_name']
        for metric, result in file_scores.iteritems():
            if metric != 'accuracy' and not self.add_metrics:
                continue
            config['y_label'] = metric.replace('_', ' ').title()
            config['file_name'] = directory + metric + '_' + file_name
            self.draw_graph(result, **config)
            if split:
                f_n = 'false_negatives'
                for key, val in result.iteritems():
                    values = {key: val}
                    if metric != f_n and f_n in file_scores.keys():
                        values['%s_%s' % (key, f_n)] = file_scores[f_n][key]
                    config['file_name'] = (directory + metric + '_' +
                                           key + '-' + file_name)
                    self.draw_graph(values, **config)

    def find_data_and_draw_graphs(self, graph_folder):
        json_files = get_json_in_subfolders(
            graph_folder, exclude_strings=['combo']
        )
        for json_file in json_files:
            x_ticks = []
            if 'days' in json_file:
                x_ticks = [1, 4, 9, 14, 19, 29, 39, 49, 59, 69, 79, 89]

            self.draw_graph_from_file(
                json_file.replace('-config.json', ''), x_ticks=x_ticks
            )

    def write_out_results(self, experiment, results, x_values, x_label,
                          y_label, file_name=None, draw=False):
        if not self.file_repl:
            date = datetime.utcnow().replace(microsecond=0).isoformat()
        else:
            date = self.file_repl
        file_name = file_name or '%s-%s-graph.png' % (experiment, date)

        config = {
            'x_values': x_values, 'file_name': file_name, 'y_label': y_label,
            'x_label': x_label,
        }

        results_list = {}
        overall_best_est = {'name': '', 'score': -1, 'obj': None}
        with open('%s-%s-info.txt' % (experiment, date), 'w') as info_file:
            for result, best_estimator in results:
                if not result:
                    continue
                if best_estimator.get('score', -1) > overall_best_est['score']:
                    overall_best_est = best_estimator
                expr_label = '%s - ' % result['stats'].pop('label')
                info_file.write(
                    expr_label + json.dumps(result['stats']) + '\n'
                )
                if 'balanced_stats' in result:
                    info_file.write(
                        '\t' + expr_label +
                        json.dumps(result['balanced_stats']) + '\n'
                    )
                for estimator, metrics in result['values']:
                    for metric, values in metrics.iteritems():
                        if metric not in results_list:
                            results_list[metric] = {}
                        if estimator in results_list[metric]:
                            results_list[metric][estimator].append(
                                values.tolist()
                            )
                        else:
                            results_list[metric][estimator] = [values.tolist()]
        joblib.dump(
            overall_best_est,
            '%s-%s-est.pkl' % (experiment, overall_best_est['name']),
        )

        with open('%s-%s-config.json' % (experiment, date), 'w') as config_f:
            config_f.write(json.dumps(dict(config, results=results_list)))

        if draw:
            for metric, metric_results in results_list.iteritems():
                if metric != 'accuracy' and not self.add_metrics:
                    continue
                agg_metric = {i: [[], []] for i in metric_results.keys()}
                for estimator, est_results in metric_results.iteritems():
                    for values in est_results:
                        values = numpy.array(values)
                        agg_metric[estimator][0].append(values.mean())
                        agg_metric[estimator][1].append(values.std())
                file_name = '%s_%s' % (metric, config['file_name'])
                y_label = metric.replace('_', ' ').title()
                metric_config = dict(
                    config, y_label=y_label, file_name=file_name
                )
                self.draw_graph(agg_metric, **metric_config)
        return dict(config, results=results_list)
