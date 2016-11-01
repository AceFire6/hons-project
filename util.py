import colorsys
import math
import re

import numpy
import pandas
import pycountry


def generate_n_rgb_colours(n, saturation=0.34, value=0.58):
    hues = numpy.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]


def get_split_and_balance(train_targets, test_targets):
    train_n = len(train_targets)
    train_tally = dict(zip(*numpy.unique(train_targets, return_counts=True)))
    test_n = None
    test_tally = {True: None, False: None}
    if test_targets is not None:
        test_n = len(test_targets)
        test_tally = dict(zip(*numpy.unique(test_targets, return_counts=True)))
    return {'train_n': train_n, 'test_n': test_n,
            'pos_train': train_tally.get(True, 0),
            'neg_train': train_tally.get(False, 0),
            'pos_test': test_tally.get(True, 0),
            'neg_test': test_tally.get(False, 0)}


def calculate_false_negatives_and_positives(y_predict, y_actual,
                                            normalize=True):
    total = len(y_predict)
    if type(y_actual) == pandas.Series:
        neg_count = sum(
            1 for i in xrange(total)
            if y_predict[i] == False and y_actual.iloc[i] == True
        )
        pos_count = sum(
            1 for i in xrange(total)
            if y_predict[i] == True and y_actual.iloc[i] == False
        )
    else:
        neg_count = sum(
            1 for i in xrange(total)
            if y_predict[i] == False and y_actual[i] == True
        )
        pos_count = sum(
            1 for i in xrange(total)
            if y_predict[i] == True and y_actual[i] == False
        )
    return {
        'positives': float(pos_count) / total if normalize else pos_count,
        'negatives': float(neg_count) / total if normalize else neg_count,
    }


def round_up(x, nearest=1):
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
            for c in pycountry.countries:
                if country in c.name:
                    codes.append(c.alpha3)
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
