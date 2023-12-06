import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt


def two_standard_deviations(p):
    return sqrt(p) * 2


def always_true(i):
    return True


def average(x, n, a):
    # computes an average from last average a and n indexed at 0 for next item x
    return (n * a + x) / (n + 1)


def loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))
