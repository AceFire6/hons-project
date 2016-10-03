import colorsys
import math

import numpy
import pycountry


def generate_n_rgb_colours(n, saturation=0.34, value=0.58):
    hues = numpy.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]


def round_up(x, nearest=1):
    x = x + 1 if x % nearest == 0 else x
    return int(nearest * math.ceil(float(x) / nearest))


def print_title(title, underline=''):
    print '%s\n%s' % (title, underline * len(title))


def list_index(indices, *select_lists):
    return_lists = list(select_lists)
    if not select_lists or not indices:
        return return_lists if len(return_lists) > 1 else return_lists[0]
    for i in range(len(select_lists)):
        return_lists[i] = [select_lists[i][index] for index in indices]
    return return_lists


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
