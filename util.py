import colorsys
import math
from warnings import filterwarnings

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


def list_index(select_list, indices):
    if not select_list or not indices:
        return select_list
    return [select_list[i] for i in indices]


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


def filter_warnings():
    # Preserve output sanity.
    # These warnings don't affect anything and are unnecessary.
    filterwarnings('ignore', 'numpy not_equal will not check object')
    filterwarnings('ignore', 'downsample module has been moved to')
