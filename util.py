import colorsys
import math

import numpy
import pycountry


def generate_n_rgb_colours(n, saturation=0.34, value=0.58):
    hues = numpy.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]


def get_split_and_balance(train_targets, test_targets):
    train_n, test_n = len(train_targets), len(test_targets)
    train_tally = train_targets.value_counts()
    test_tally = test_targets.value_counts()
    return {'train_n': train_n, 'test_n': test_n,
            'pos_train': train_tally[True], 'neg_train': train_tally[False],
            'pos_test': test_tally[True], 'neg_test': test_tally[False]}


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
    return codes
