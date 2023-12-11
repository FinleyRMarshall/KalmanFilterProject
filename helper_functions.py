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


class Satellite:
    def __init__(self, a=(pi * 2) / 360, h=10, q=2, r=1, radius=10, start_angle=None):
        self.a = a
        self.h = h
        self.radius = radius
        self.r = r
        self.q = q
        self.x1 = []
        self.x2 = []
        self.times = []
        self.measurements_times = []
        self.current_time = 0
        self.start_angle = start_angle
        self.measurements = []

    def __random_start__(self):
        #returns a random start cords for both x1 and x2
        #if the start angle is not given
        if self.start_angle is None:
            self.start_angle = np.random.uniform(2 * pi)

        x1_cord = self.radius * cos(self.start_angle)
        x2_cord = self.radius * sin(self.start_angle)
        return x1_cord, x2_cord

    def next_cord(self, receive=True):
        self.times.append(self.current_time)

        if len(self.x1) == 0:
            x1_cord, x2_cord = self.__random_start__()
            self.x1.append(x1_cord)
            self.x2.append(x2_cord)

            x1_measurement = x1_cord + np.random.normal(0, self.r)
            self.measurements.append(x1_measurement)
            self.measurements_times.append(self.current_time)

            self.current_time += self.h
            return x1_measurement

        trans_matrix = np.array([[cos(self.a * self.h), -sin(self.a * self.h)], [sin(self.a * self.h), cos(self.a * self.h)]])
        last_cords = np.array([self.x1[-1], self.x2[-1]])

        cords = np.dot(trans_matrix, last_cords)
        x1_cord, x2_cord = cords
        x1_noise, x2_noise = np.random.normal(0, self.q, 2)
        x1_cord += x1_noise
        x2_cord += x2_noise

        self.x1.append(x1_cord)
        self.x2.append(x2_cord)

        if receive:
            x1_measurement = x1_cord + np.random.normal(0, self.r)
            self.measurements.append(x1_measurement)
            self.measurements_times.append(self.current_time)
        else:
            x1_measurement = None

        self.current_time += self.h
        return x1_measurement if receive else None

    def graph(self, show=True):
        #graphs x1, x2 true cords and the measurement of x1
        plt.title('Dimensions X1, X2 and X2 measurements of the satellite over time')
        plt.plot(self.measurements_times, self.measurements, '.', label='Measurements')
        plt.plot(self.times, self.x1, label='x1')
        plt.plot(self.times, self.x2, label='x2')
        if show:
            plt.legend()
            plt.show()

"""
loop_count = loop_size(h, a, revolutions)


satellite = Satellite(a=a, h=h, q=q, r=r, radius=radius, start_angle=start_angle)

prediction_data = []
estimate_data = []

for i in range(loop_count):
    receive = receive_function(i)

    z = satellite.next_cord(receive=receive)

    x1, x2 = kf.predict()
    prediction_data.append((x1, x2, kf.p))

    if z is not None:
        x1, x2 = kf.update(z)
        estimate_data.append((x1, x2, kf.p))

return satellite, prediction_data, estimate_data

"""

def satellite_Analysis(satellites, revolutions, parameter, values, receive_function):

    for value in values:
        radius = 10
        satellite_parameters = {}
        a = (pi * 2) / 360
        satellite_parameters['h'] = 10
        satellite_parameters['r'] = 2
        satellite_parameters['q'] = 0.1

        satellite_parameters[parameter] = value

        h = satellite_parameters['h']
        r = satellite_parameters['r']
        q = satellite_parameters['q']



        for j in satellites
            X = np.array([0, 0])
            F = np.array([[1, -a * h], [h * a, 1]])
            H = np.array([1, 0]).reshape(1, 2)
            Q = np.array([[q, 0], [0, q]])
            R = np.array([r]).reshape(1, 1)
            P = np.array([[radius, 0], [0, radius]])

            kf = KalmanFilter(f=F, h=H, q=Q, r=R, x=X, p=P)
            s = Satellite(a=a, h=h, q=q, r=r, radius=radius)



satellite_Analysis(3, 3, 'h', [5, 6], always_true)