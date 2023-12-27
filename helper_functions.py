import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt
from Graph_functions import *
from main import *


def three_standard_deviations(p):
    return sqrt(p) * 3


def always_true(i):
    return True


def average(x, n, a):
    # computes an average from last average a and n indexed at 0 for next item x
    return (n * a + x) / (n + 1)


def loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))


def satellite_analysis(satellites, revolutions, parameter, values, receive_values=None):
    if receive_values == None:
        receive_values = [always_true for i in range(len(values))]
    data = []

    for num_value, value in enumerate(values):
        radius = 10
        a = (pi * 2) / 360

        satellite_parameters = {}
        satellite_parameters['h'] = 10
        satellite_parameters['r'] = 2
        satellite_parameters['q'] = 0.1

        satellite_parameters[parameter] = value

        h = satellite_parameters['h']
        r = satellite_parameters['r']
        q = satellite_parameters['q']

        x1_average_abs_error = []
        x2_average_abs_error = []
        measurement_average_abs_error = []

        objects = []

        for i in range(satellites):
            X = np.array([0, 0])
            F = np.array([[1, -a * h], [h * a, 1]])
            H = np.array([1, 0]).reshape(1, 2)
            Q = np.array([[q, 0], [0, q]])
            R = np.array([r]).reshape(1, 1)
            P = np.array([[radius, 0], [0, radius]])

            kf = KalmanFilter(f=F, h=H, q=Q, r=R, x=X, p=P)
            s = Satellite(a=a, h=h, q=q, r=r, radius=radius)

            objects.append((kf, s))


        for j in range(loop_size(h, a, revolutions)):
            x1_error = 0
            x2_error = 0
            measurement_error = 0
            receive = receive_values[num_value](j)

            for num, i in enumerate(objects):
                kf, s = i
                z = s.next_cord(receive=receive)

                x1, x2 = kf.predict()

                if z is not None:
                    x1, x2 = kf.update(z)
                    measurement_error += abs(s.x1[-1] - z)

                x1_error += abs(s.x1[-1] - x1)
                x2_error += abs(s.x2[-1] - x2)

            if receive:
                measurement_average_abs_error.append(measurement_error / satellites)
            x1_average_abs_error.append(x1_error / satellites)
            x2_average_abs_error.append(x2_error / satellites)

        data.append((measurement_average_abs_error, x1_average_abs_error, x2_average_abs_error, s.times[:],
                     s.measurements_times[:], value))

    graph_analysis(data, parameter)
