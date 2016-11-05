import argparse
import itertools
import math
import matplotlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from experiment_data import ExperimentData
from experiment_helpers import ExperimentHelper, draw_table
from util import generate_n_rgb_colours, get_short_codes, print_title


def all_data_performance_experiment():
    """All Data Performance Experiment"""
    print_title('All Data Performance Experiment', '-')
    feature_data = chw_data.get_features()
    target_data = chw_data.get_targets()
    result_scores = exp.param_run(feature_data, target_data)
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
        result_scores = exp.param_run(
            chw_data.features, chw_data.targets, debug_label=feature_name,
        )
        out_results.append(result_scores)
        chw_data.drop_feature(feature_name)

    x_values = ['None'] + other_features.keys()
    exp.write_out_results(
        'features', out_results, x_values, 'Feature', 'Accuracy',
        draw=args.graph,
    )


def modified_features_experiment():
    """Effect of Modifying Features"""
    def project_normalize(dataset, feature_set):
        ds = dataset.copy()
        feature_set = [i for i in feature_set if not i.startswith('nX')]
        x_nums = ['X%s' % i for i in range(1, 91)]
        grouped = ds[['projectCode'] + x_nums].groupby('projectCode')
        g_normalized = grouped.apply(
            lambda g: g[x_nums] / math.sqrt((g[x_nums] ** 2).sum(axis=1).sum())
        )
        for col in x_nums:
            ds[col] = g_normalized[col]
        return ds[x_nums + feature_set]

    print_title('Running Modified Features Experiment', '-')

    labels = []
    out_results = []
    features = chw_data.feature_labels
    modified_features = {
        'Binarized': lambda x: x[features] > 0,
        'Project_Normalize': (
            lambda x: project_normalize(x, features)
        ),
    }

    for name, modifier in modified_features.iteritems():
        feature_data = modifier(chw_data.dataset)
        target_data = chw_data.get_targets()
        result_scores = exp.param_run(feature_data, target_data)
        labels.append(name)
        out_results.append(result_scores)
    exp.write_out_results(
        'modify', out_results, labels, 'Modifier', 'Accuracy', draw=args.graph,
    )


def effect_of_day_data_experiment():
    """Effect of Number of Days Included (1-90)"""
    print_title('Running Effect of Day Experiment', '-')
    out_results = []
    # Go through all values of X (1-90)
    x_val_range = args.x_ticks if args.x_ticks else range(1, 91)
    for x in x_val_range:
        x_col = ['nX%d' % i for i in range(x + 1, 91)]
        feature_data = chw_data.get_features(drop_cols=chw_data.categories)
        feature_data = feature_data.drop(x_col, axis=1)
        target_data = chw_data.get_targets()
        result_scores = exp.param_run(
            feature_data, target_data, debug_label=x, cross_folds=10,
        )
        out_results.append(result_scores)
    exp.write_out_results(
        'days', out_results, x_val_range, 'Number of days included',
        'Accuracy', draw=args.graph
    )


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
            result_scores = exp.param_run(
                feature_data, target_data,
                test_features=test_features, test_targets=test_targets,
                debug_label=country, repeat_test=True,
            )
        else:
            result_scores = exp.param_run(
                test_features, test_targets,
                test_features=feature_data, test_targets=target_data,
                debug_label=country, repeat_test=True,
            )
        out_results.append(result_scores)
    countries = get_short_codes(countries)
    experiment = 'country' + ('_inverse' if inverse else '')
    exp.write_out_results(
        experiment, out_results, countries, 'Country', 'Accuracy',
        draw=args.graph,
    )


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
            result_scores = exp.param_run(
                feature_data, target_data,
                test_features=test_features, test_targets=test_targets,
                debug_label=sector, repeat_test=True,
            )
        else:
            result_scores = exp.param_run(
                test_features, test_targets,
                test_features=feature_data, test_targets=target_data,
                debug_label=sector, repeat_test=True,
            )
        out_results.append(result_scores)

    experiment = 'sector' + ('_inverse' if inverse else '')
    exp.write_out_results(
        experiment, out_results, sectors, 'Sector', 'Accuracy',
        draw=args.graph,
    )


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
            result_scores = exp.param_run(
                feature_data, target_data,
                test_features=test_features, test_targets=test_targets,
                debug_label=project_code, repeat_test=True
            )
        else:
            result_scores = exp.param_run(
                test_features, test_targets,
                test_features=feature_data, test_targets=target_data,
                debug_label=project_code, repeat_test=True
            )

        out_results.append(result_scores)

    experiment = 'project' + ('_inverse' if inverse else '')
    exp.write_out_results(
        experiment, out_results, map(str, project_codes.keys()),
        'Project', 'Accuracy', draw=args.graph,
    )


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

    projects = []
    for project_code in project_codes:
        col_select = {'projectCode': project_code}

        test_features = chw_data.get_features(col_select)
        test_targets = chw_data.get_targets(col_select)

        test_count = test_targets.value_counts().values
        if any(test_count < 50):
            print 'Skipping project %d: Too few values' % project_code
            print test_count
            continue

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

        both_count_too_small = (
            any(same_country[1].value_counts().values < 50) or
            any(same_sector[1].value_counts().values < 50)
        )
        if both_count_too_small:
            print 'Skipping project %d: Too few values' % project_code
            print test_count
            continue

        projects.append(project_code)

        training_sets = {'All Data': all_data, 'Same Country': same_country,
                         'Same Sector': same_sector,
                         'Same Country & Sector': same_country_and_sector}
        for name in training_order:
            label = '%d - %s' % (project_code, name)
            train_features, train_targets = training_sets[name]
            train_count = train_targets.value_counts().values
            if all(train_count > 50):
                result_scores = exp.param_run(
                    train_features, train_targets,
                    test_features=test_features, test_targets=test_targets,
                    debug_label=label, repeat_test=True,
                )
            else:
                result_scores = {'stats': {'label': label}, 'values': ()}
                print 'Too few classes'
            out_results.append(result_scores)
    exp.write_out_results('combo', out_results, projects, None, 'Accuracy')


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
        modified_features_experiment,
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
    parser.add_argument('-G', '--graph-all', dest='graph_folder',
                        help=('Generate graphs for each *-config.json'
                              ' in the folder specified'))
    parser.add_argument('-T', '--draw-table', dest='table_file',
                        help='Generate HTML table from the specified file')
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
    svm = SVC(C=100, class_weight='balanced', cache_size=1000)
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

    exp = ExperimentHelper(estimators, args, colours, styles)

    if args.list:
        print_title('All Experiments:', '-')
        for i in range(len(experiment_functions)):
            print '%d. %s' % (i, experiment_functions[i].__doc__)
    elif args.graph:
        exp.draw_graph_from_file(args.graph, args.split, args.x_ticks)
    elif args.graph_folder:
        exp.find_data_and_draw_graphs(args.graph_folder)
    elif args.table_file:
        draw_table(args.table_file)
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
