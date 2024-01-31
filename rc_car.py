import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt
from Graph_functions import *
from measurement_functions import *


class RC_Car_EKF():
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


def rc_car_example(h=1, k=0.1, r=60, loop_count=3028, control_file='controls/track_controls.txt',
                   receive_function=every_second, alpha_variance=0.1, theta_variance=0.01):
    F = np.array([[1, 0, h, 0], [0, 1, 0, h], [0, 0, 1 - k * h, 0], [0, 0, 0, 1 - k * h]])
    X = np.array([0, 0, 1, 0])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    P = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.array([[r ** 2, 0], [0, r ** 2]])
    Q = np.array([[.01, 0, 0, 0], [0, .01, 0, 0], [0, 0, .01, 0], [0, 0, 0, .01]])

    kf = RC_Car_EKF(f=F, h=H, q=Q, r=R, p=P, x=X, time_step=h)
    car = RC_car(k=k, h=h, r=r, control_file=control_file, alpha_variance=alpha_variance,
                 theta_variance=theta_variance)

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


def car_position_analysis(cars, txt, title):
    measurement_distance_data = []
    gps_distance_data = []

    for i in range(cars):
        car, car_data = rc_car_example()
        measurement_distance, model_distance = graph_car(car, car_data, txt, output='0')
        measurement_distance_data += measurement_distance
        gps_distance_data += model_distance

    graph_boxplot([measurement_distance_data, gps_distance_data], ['GPS Measurements', 'Estimates'], txt, title)


def car_velocity_analysis(cars, txt, title):
    v1_distance = []
    v2_distance = []

    for i in range(cars):
        car, car_data = rc_car_example()
        model_v1_distance, model_v2_distance = graph_car(car, car_data, txt, output='1')
        v1_distance += model_v1_distance
        v2_distance += model_v2_distance

    graph_boxplot([v1_distance, v2_distance], ['V1 Estimates', 'V2 Estimates'], txt, title)