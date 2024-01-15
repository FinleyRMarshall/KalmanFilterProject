import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt
from Graph_functions import *

class Satellite:
    def __init__(self, a=1, h=1, q=0, r=0, radius=10, start_angle=None):
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
            plt.legend(loc='upper left')
            plt.show()

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

def loop_size(h, a, revolution):
    return int((360 // h * revolution) / (a * 360 / (2 * pi)))

def satellite_analysis(satellites, revolutions, parameter, values, receive_values=None):
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

    graph_analysis(data, parameter)

def rc_car_control_input_model(x,  h, theta, alpha):
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
        self.x = np.dot(self.f, self.x) + rc_car_control_input_model(self.x, self.time_step, theta, alpha)
        self.p = np.dot(np.dot(self.f, self.p), self.f.T) + self.q
        return self.x

    def update(self, z):
        y = z - np.dot(self.h, self.x)
        s = np.dot(np.dot(self.h, self.p), self.h.T) + self.r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))
        self.x = self.x + np.dot(k,y)
        self.p = np.dot(np.identity(len(self.x)) - np.dot(k, self.h), self.p)
        return self.x



