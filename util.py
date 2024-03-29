import colorsys
import math
import os
import re

import numpy
import pandas
import pycountry


def generate_n_rgb_colours(n, saturation=0.34, value=0.58):
    hues = numpy.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]


def get_split_and_balance(train_targets, test_targets):
    """Use the test and training targets to determine the size of each set
    and calculate the balance of each dataset.

    :return:Dictionary with the test and train splits and balance information.
    """
    train_n = len(train_targets)
    train_tally = dict(zip(*numpy.unique(train_targets, return_counts=True)))
    test_n = None
    test_tally = {True: None, False: None}
    if test_targets is not None:
        test_n = len(test_targets)
        test_tally = dict(zip(*numpy.unique(test_targets, return_counts=True)))
    return {
        'train_n': train_n,
        'test_n': test_n,
        'pos_train': train_tally.get(True, 0),
        'neg_train': train_tally.get(False, 0),
        'pos_test': test_tally.get(True, 0),
        'neg_test': test_tally.get(False, 0)
    }


def calculate_false_negatives_and_positives(y_predict, y_actual):
    """Calculates the false negative and positive rates given the predicted
    values and the actual values.

    :return:Dictionary with the false positive and negative rates.
    """
    total = len(y_predict)
    if type(y_actual) == pandas.Series:
        neg_count = sum(
            1 for i in xrange(total)
            # predicted false, actually true
            if not y_predict[i] and y_actual.iloc[i]
        )
        pos_count = sum(
            1 for i in xrange(total)
            # predicted true, actually false
            if y_predict[i] and not y_actual.iloc[i]
        )
    else:
        neg_count = sum(
            1 for i in xrange(total)
            # predicted false, actually true
            if not y_predict[i] and y_actual[i]
        )
        pos_count = sum(
            1 for i in xrange(total)
            # predicted true, actually false
            if y_predict[i] and not y_actual[i]
        )
    return {
        'positives': float(pos_count) / total,
        'negatives': float(neg_count) / total,
    }


def round_up(x, nearest=1):
    """Custom rounding method. Supports rounding to the nearest value defined
     by the `nearest` parameter.

    :return:integer value of x rounded to the nearest value specified by
    `nearest`.
    """
    x = x + 1 if x % nearest == 0 else x
    return int(nearest * math.ceil(float(x) / nearest))


def print_title(title, underline='', format_dict=None):
    if format_dict is not None:
        title = title.format(format_dict)
    print '%s\n%s' % (title, underline * len(title))


def list_index(indices, *select_lists):
    return_lists = list(select_lists)
    if select_lists and indices:
        for i in range(len(select_lists)):
            return_lists[i] = [select_lists[i][index] for index in indices]
    return return_lists if len(return_lists) > 1 else return_lists[0]


def get_short_codes(countries):
    codes = []
    for country in countries:
        country = country.title()
        try:
            codes.append(pycountry.countries.get(name=country).alpha3)
        except KeyError:
            for country_info in pycountry.countries:
                if country in country_info.name:
                    codes.append(country_info.alpha3)
                    continue
            # Not a country - therefore Not Specified
            if country == 'Not Specified':
                codes.append('NS')
    return codes


def replace_isodate(file_name, replace_str):
    if not replace_str:
        return file_name
    date_time_regex = r'(?:\d{2,4}-?){3}T(?:\d{2,4}:?){3}'
    if re.search(date_time_regex, file_name):
        return re.sub(date_time_regex, replace_str, file_name)
    return file_name


def get_json_in_subfolders(parent_folder, exclude_strings=list()):
    """Recursively walk the subfolders of the supplied parent folder looking
     for json files. Excludes any files with names containing values specified
     in `exclude_strings`.

    :return:list of paths to json files that meet the criteria.
    """
    if not os.path.isdir(parent_folder):
        return None

    file_list = os.listdir(parent_folder)
    json_files = map(
        lambda f_json: os.path.join(parent_folder, f_json),  # Complete path
        filter(lambda x: x.endswith('.json'), file_list),  # Filter on .json
    )
    subfolders = filter(
        lambda x: os.path.isdir(os.path.join(parent_folder, x)), file_list
    )

    if subfolders:
        for subfolder in subfolders:
            json_files += get_json_in_subfolders(
                os.path.join(parent_folder, subfolder), exclude_strings
            )

    for exclude_string in exclude_strings:
        json_files = filter(lambda x: exclude_string not in x, json_files)

    return json_files
