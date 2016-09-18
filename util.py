import colorsys
import math
from warnings import filterwarnings

import numpy


def generate_n_rgb_colours(n, saturation=0.34, value=0.58):
    hues = numpy.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]


def round_up(x, nearest=1):
    x = x + 1 if x % nearest == 0 else x
    return int(nearest * math.ceil(float(x) / nearest))


def filter_warnings():
    # Preserve output sanity.
    # These warnings don't affect anything and are unnecessary.
    filterwarnings('ignore', 'numpy not_equal will not check object')
    filterwarnings('ignore', 'downsample module has been moved to')
