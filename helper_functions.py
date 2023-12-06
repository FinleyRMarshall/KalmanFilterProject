import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt

def two_standard_deviations(p):
    return sqrt(p) * 2


def always_true(i):
    return True


def loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))