import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt
from Graph_functions import *
from measurement_functions import *


class KalmanFilter(object):
    def __init__(self, f, h, q, r, p, x, b=0, u=0):
        self.f = f
        self.h = h
        self.q = q
        self.r = r
        self.p = p
        self.x = x
        self.b = b
        self.u = u
        self.n = f.shape[1]

    def predict(self):
        self.x = np.dot(self.f, self.x) + np.dot(self.b, self.u)
        self.p = np.dot(np.dot(self.f, self.p), self.f.T) + self.q
        return self.x

    def update(self, z):
        y = z - np.dot(self.h, self.x)
        s = np.dot(np.dot(self.h, self.p), self.h.T) + self.r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))
        self.x = self.x + np.dot(k,y)
        self.p = np.dot(np.identity(len(self.x)) - np.dot(k, self.h), self.p)
        return self.x


class Satellite:
    def __init__(self, a=None, h=None, q=None, r=None, radius=None, start_angle=None):
        # Creates the satellite, given all the required values as described before.
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
        # The start angle is from the x1 positive axis, anti-clockwise
        self.measurements = []

    def __random_start__(self):
        # Returns a random start position for the satellite
        # If the start angle is not given
        if self.start_angle is None:
            self.start_angle = np.random.uniform(2 * pi)

        x1_cord = self.radius * cos(self.start_angle)
        x2_cord = self.radius * sin(self.start_angle)
        return x1_cord, x2_cord

    def next_cord(self, receive=True):
        self.times.append(self.current_time)

        if len(self.x1) == 0:
            # If the satellite has just been created.
            x1_cord, x2_cord = self.__random_start__()
            self.x1.append(x1_cord)
            self.x2.append(x2_cord)

            x1_measurement = x1_cord + np.random.normal(0, self.r)
            self.measurements.append(x1_measurement)
            self.measurements_times.append(self.current_time)

            self.current_time += self.h
            # The model only sees the x1 measurement
            return x1_measurement

        # Uses the transition matrix to find the next position of the satellite
        # Noise is then added, to simulate the random movement
        trans_matrix = np.array([[cos(self.a * self.h), -sin(self.a * self.h)],
                                 [sin(self.a * self.h), cos(self.a * self.h)]])
        last_cords = np.array([self.x1[-1], self.x2[-1]])
        x1_cord, x2_cord = np.dot(trans_matrix, last_cords)
        x1_noise, x2_noise = np.random.normal(0, self.q, 2)
        x1_cord += x1_noise
        x2_cord += x2_noise
        self.x1.append(x1_cord)
        self.x2.append(x2_cord)

        # If the model is to receive a measurement, process the measurement
        if receive:
            x1_measurement = x1_cord + np.random.normal(0, self.r)
            self.measurements.append(x1_measurement)
            self.measurements_times.append(self.current_time)
        else:
            x1_measurement = None

        self.current_time += self.h

        # The model only sees the x1 measurement
        return x1_measurement if receive else None


    def graph(self, show=True, title=None, figure_txt='', measurements = True):
        # Graphs x1, x2 true cords and the measurements of x1
        # Subplot allows for two graphs side by side. Set show = False for the first graph
        if title is None:
            title = 'True Values of $X_1$, $X_2$ and Measurements of $X_1$'
        if measurements:
            plt.plot(self.measurements_times, self.measurements, '.', label='Measurements')
        plt.plot(self.times, self.x1, label='$X_1$')
        plt.plot(self.times, self.x2, label='$X_2$')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("$X_1$ and $X_2$")
        plt.legend(loc='upper left')

        if show:
            plt.figtext(0.5, -0.10, figure_txt, wrap=True, horizontalalignment='center', fontsize=12)
            plt.show()


def satellite_example(h=10, a=(pi * 2)/ 360, r=2, q=0.1, radius=10, revolutions=2, receive_function=always_true, start_angle=None):
    # Calculates how many times to run the model to get the desired number of revolutions
    loop_count = satellite_loop_size(h, a, revolutions)

    # The same values as explained before
    X = np.array([0, 0])
    F = np.array([[1, -a * h], [h * a, 1]])
    H = np.array([1, 0]).reshape(1, 2)
    Q = np.array([[q, 0], [0, q]])
    R = np.array([r])
    P = np.array([[radius, 0], [0, radius]])

    kf = KalmanFilter(f=F, h=H, q=Q, r=R, x=X, p=P)
    satellite = Satellite(a=a, h=h, q=q, r=r, radius=radius, start_angle=start_angle)

    prediction_data = []
    estimate_data = []

    # Run the model
    for i in range(loop_count):
        receive = receive_function(i)

        z = satellite.next_cord(receive=receive)

        x1, x2 = kf.predict()
        prediction_data.append((x1, x2, kf.p))

        if z is not None:
            x1, x2 = kf.update(z)
            estimate_data.append((x1, x2, kf.p))

    return satellite, prediction_data, estimate_data


def satellite_analysis(satellites, revolutions, parameter, values, figure_txt, title=None, receive_values=None):
    # Runs the satellite analysis, quite similar to satellite_example
    if receive_values == None:
        receive_values = [always_true for i in range(len(values))]
    data = []

    for num_value, value in enumerate(values):
        # Assigning the values for the satellites
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

        # For graphing
        x1_average_abs_error = []
        x2_average_abs_error = []
        measurement_average_abs_error = []

        objects = []

        # Create all the satellites
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

        # Run the model
        for j in range(satellite_loop_size(h, a, revolutions)):
            x1_error = 0
            x2_error = 0
            measurement_error = 0
            receive = receive_values[num_value](j)
            # For each satellite, and model for each satellite
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

    graph_analysis(data, parameter, figure_txt, title=title)


def satellite_position_analysis(satellites, revolutions,txt, title, parameter='h', value=10, receive_function=always_true):
    measurement_distance_data = []
    error_data = []

    satellite_parameters = {}
    satellite_parameters['h'] = 10
    satellite_parameters['r'] = 2
    satellite_parameters['q'] = 0.1

    satellite_parameters[parameter] = value

    h = satellite_parameters['h']
    r = satellite_parameters['r']
    q = satellite_parameters['q']

    for i in range(satellites):
        satellite, prediction_data, estimate_data = satellite_example(h=h, r=r, q=q, revolutions = revolutions, receive_function=receive_function)
        measurement_error, x1_error, x2_error = graph_error(satellite, prediction_data, estimate_data, return_data = True)
        measurement_distance_data += measurement_error
        error_data += x1_error
        error_data += x2_error

    graph_boxplot([measurement_distance_data, error_data], ['Measurements', 'Estimates'], txt, title)



def satellite_loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))