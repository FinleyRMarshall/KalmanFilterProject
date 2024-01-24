import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt
from Graph_functions import *


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


def always_true(i):
    return True


def always_false(i):
    return False


def every_second(i):
    return i % 2 == 0


def loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))


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
        for j in range(loop_size(h, a, revolutions)):
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


def rc_car_control_input_model(x,  h, alpha, theta):
    # Controls for the rc car
    # The explanation for this function can be found in the control input model section
    x_1, x_2, v_1, v_2 = x
    mag_v = sqrt(v_1**2 + v_2**2)
    scalar_factor = 1 + (alpha * h)/(mag_v)
    new_x_1 = h/2*(-v_1 + scalar_factor*(v_1*cos(theta) - v_2*sin(theta)))
    new_x_2 = h/2*(-v_2 + scalar_factor*(v_1*sin(theta) + v_2*cos(theta)))
    new_v_1 = scalar_factor*(v_1*cos(theta) - v_2*sin(theta)) - v_1
    new_v_2 = scalar_factor*(v_1*sin(theta) + v_2*cos(theta)) - v_2
    return np.array([new_x_1, new_x_2, new_v_1, new_v_2])


class RC_Car_EKF(object):
    def __init__(self, f, h, q, r, p, x, time_step):
        self.f = f
        self.h = h
        self.q = q
        self.r = r
        self.p = p
        self.x = x
        self.time_step = time_step
        self.n = f.shape[1]

    def predict(self, alpha, theta):
        self.x = np.dot(self.f, self.x) + rc_car_control_input_model(self.x, self.time_step, alpha, theta)
        self.p = np.dot(np.dot(self.f, self.p), self.f.T) + self.q
        return self.x

    def update(self, z):
        y = z - np.dot(self.h, self.x)
        s = np.dot(np.dot(self.h, self.p), self.h.T) + self.r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))
        self.x = self.x + np.dot(k, y)
        self.p = np.dot(np.identity(len(self.x)) - np.dot(k, self.h), self.p)
        return self.x


class RC_car:
    def __init__(self, k=0.1, h=1, r=60, alpha_variance=0.1, theta_variance=0.01, control_file=None):
        self.k = k  # Drag
        self.h = h  # Time step
        self.r = r
        self.alpha_variance = alpha_variance
        self.theta_variance = theta_variance

        self.x1 = [0]
        self.x2 = [0]
        self.v1 = [1]
        self.v2 = [0]

        self.times = [0]
        self.current_time = 0

        self.x1_measurements = []
        self.x2_measurements = []
        self.measurements_times = []

        self.file = open(control_file, 'r')

    def close_file(self):
        self.file.close()

    def next_movement(self, receive):
        self.current_time += self.h
        self.times.append(self.current_time)

        controls = self.file.readline()
        if controls == '':  # End of file, no more control inputs
            controls = '0 0'
        alpha, theta = controls.split()
        alpha = float(alpha)
        theta = float(theta)
        f = np.array([[1, 0, self.h, 0], [0, 1, 0, self.h], [0, 0, 1 - self.k * self.h, 0], [0, 0, 0, 1 - self.k * self.h]])
        x = np.array([self.x1[-1], self.x2[-1], self.v1[-1], self.v2[-1]])
        x1, x2, v1, v2 = np.dot(f, x) + rc_car_control_input_model(x, self.h, alpha, theta)

        self.x1.append(x1)
        self.x2.append(x2)
        self.v1.append(v1)
        self.v2.append(v2)

        # If the model is to receive a measurement, process the measurement, else give None
        x1_measurement = None
        x2_measurement = None
        if receive:
            x1_measurement = x1 + np.random.normal(0, self.r)
            x2_measurement = x2 + np.random.normal(0, self.r)
            self.x1_measurements.append(x1_measurement)
            self.x2_measurements.append(x2_measurement)
            self.measurements_times.append(self.current_time)

        return x1_measurement, x2_measurement, alpha + np.random.normal(0,self.alpha_variance), theta + np.random.normal(0, self.theta_variance)


def rc_car_example(h=1, k=0.1, r=60, loop_count=3028, control_file='controls/track_controls.txt', receive_function=every_second, alpha_variance=0.1, theta_variance=0.01):
    F = np.array([[1, 0, h, 0], [0, 1, 0, h], [0, 0, 1 - k * h, 0], [0, 0, 0, 1 - k * h]])
    X = np.array([0, 0, 1, 0])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    P = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.array([[r ** 2, 0], [0, r ** 2]])
    Q = np.array([[.01, 0, 0, 0], [0, .01, 0, 0], [0, 0, .01, 0], [0, 0, 0, .01]])

    kf = RC_Car_EKF(f=F, h=H, q=Q, r=R, p=P, x=X, time_step=h)
    car = RC_car(k=k, h=h, r=r, control_file=control_file, alpha_variance=alpha_variance, theta_variance=theta_variance)

    car_data = [(kf.x, kf.p)]

    for i in range(loop_count):
        receive = receive_function(i)
        x1, x2, alpha, theta = car.next_movement(receive)

        kf.predict(alpha, theta)
        if x1 is not None:
            kf.update(np.array([x1, x2]))

        car_data.append((kf.x, kf.p))

    car.close_file()
    return car, car_data
