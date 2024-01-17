import matplotlib.pyplot as plt
import numpy as np
from helper_functions import *


def average(x, n, a):
    # Computes an average from last average a and n indexed at 0 for next item x
    return (n * a + x) / (n + 1)

def three_standard_deviations(p):
    return sqrt(p) * 3

def graph_x1(satellite, prediction_data, estimate_data):
    # Graphs the x1 true values, measurements and the estimate
    # Graph_x1_and_p is used
    estimate_x1 = []
    predictions_x1 = []

    for num in range(len(prediction_data)):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]

        predictions_x1.append(x1_prediction)

    for num in range(len(estimate_data)):
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]

        estimate_x1.append(x1_estimate)

    plt.title('Dimension X1: Measurements, Prediction and Estimate of X1 over time')
    plt.plot(satellite.measurements_times, satellite.measurements, '.', label='Measurements')
    plt.plot(satellite.times, satellite.x1, label='X1')
    plt.plot(satellite.measurements_times, estimate_x1, label='Estimate_x1')
    #plt.plot(satellite.times, predictions_x1, '+', label='Predictions_x1')
    plt.legend(loc='upper left')
    plt.show()


def graph_x1_and_p(satellite, prediction_data, estimate_data):
    # Graphs true x1, estimate, and confidence interval
    # Given the satellite object and data from the model
    estimate_x1 = []
    predictions_x1 = []
    p_above = []
    p_below = []
    offset = 0

    for num, time in enumerate(satellite.times):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        predictions_x1.append(x1_prediction)

        if time in satellite.measurements_times:
            x1_estimate, x2_estimate, p_estimate = estimate_data[num - offset]
        else:
            offset += 1
            x1_estimate, x2_estimate, p_estimate = x1_prediction, x2_prediction, p_prediction

        p = p_estimate[0][0]
        estimate_x1.append(x1_estimate)
        p_above.append(x1_estimate + three_standard_deviations(p))
        p_below.append(x1_estimate - three_standard_deviations(p))

    plt.title('True Values vs Estimate Values of X1')
    plt.plot(satellite.times, satellite.x1, label='True X1')
    plt.plot(satellite.times, estimate_x1, label='Estimate of X1')
    plt.plot(satellite.times, p_above, label='CI Upper Bound')
    plt.plot(satellite.times, p_below, label='CI Lower Bound')

    plt.legend(loc='upper left')
    plt.show()


def graph_x2(satellite, prediction_data, estimate_data):
    # Graphs the x1 true values and the estimate
    # Graph_x2_and_p is used
    estimate_x2 = []
    predictions_x2 = []

    for num in range(len(prediction_data)):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        predictions_x2.append(x2_prediction)

    for num in range(len(estimate_data)):
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]
        estimate_x2.append(x2_estimate)

    plt.title('Dimension X2: Measurements, Estimate and P of X2 over time')
    plt.plot(satellite.times, satellite.x2, label='X2')
    plt.plot(satellite.measurements_times, estimate_x2, label='Estimate_x2')
    #plt.plot(satellite.times, predictions_x2, '+', label='Predictions_x2')
    plt.legend(loc='upper left')
    plt.show()


def graph_x2_and_p(satellite, prediction_data, estimate_data):
    # Graphs true x2, estimate, and confidence interval
    # Given the satellite object and data from the model
    estimate_x2 = []
    predictions_x2 = []
    p_above = []
    p_below = []

    offset = 0

    for num, time in enumerate(satellite.times):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        predictions_x2.append(x2_prediction)

        if time in satellite.measurements_times:
            x1_estimate, x2_estimate, p_estimate = estimate_data[num - offset]
        else:
            offset += 1
            x1_estimate, x2_estimate, p_estimate = x1_prediction, x2_prediction, p_prediction


        p = p_estimate[1][1]
        estimate_x2.append(x2_estimate)
        p_above.append(x2_estimate + three_standard_deviations(p))
        p_below.append(x2_estimate - three_standard_deviations(p))

    plt.title('True Values vs Estimate Values of X2')
    plt.plot(satellite.times, satellite.x2, label='X2')
    plt.plot(satellite.times, estimate_x2, label='Estimate of X2')
    plt.plot(satellite.times, p_above, label='CI Upper Bound')
    plt.plot(satellite.times, p_below, label='CI Lower Bound')

    plt.legend(loc='upper left')
    plt.show()


def graph_analysis(data, parameter):
    # Graphs the analysis data given from satellite_analysis in helper functions
    plt.title("Analysis of {} different values for {}".format(len(data), parameter))
    symbols = ['.', 'P', '^']

    for num, i in enumerate(data):
        measurements, x1s, x2s, times, measurements_times, value = i
        plt.plot(measurements_times, measurements, symbols[num % 3] + 'b', label = 'Measurements for {} = {}'.format(parameter, value))
        plt.plot( times, x1s, symbols[num % 3] + 'g', label = 'X1 for {} = {}'.format(parameter, value))
        plt.plot( times, x2s, symbols[num % 3] + 'r', label = 'X2 for {} = {}'.format(parameter, value))

    plt.legend(loc='upper left')
    plt.show()

def graph_error(satellite, prediction_data, estimate_data):
    # Graphs the difference of measurements, x1 estimate and x2 estimates from their true values.
    x1_error = []
    x2_error = []
    measurement_error = []
    offset = 0

    for num, time in enumerate(satellite.times):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        true_x1 = satellite.x1[num]
        true_x2 = satellite.x2[num]

        if time in satellite.measurements_times:
            x1_estimate, x2_estimate, p_estimate = estimate_data[num - offset]
            measurement_error.append(true_x1 - satellite.measurements[num - offset])
        else:
            offset += 1
            x1_estimate, x2_estimate, p_estimate = x1_prediction, x2_prediction, p_prediction


        x1_error.append(true_x1 - x1_estimate)
        x2_error.append(true_x2 - x2_estimate)

    plt.title('Error of Measurements and X1, X2')
    plt.plot(satellite.measurements_times, measurement_error, label='Measurement')
    plt.plot(satellite.times, x1_error, label='X1')
    plt.plot(satellite.times, x2_error, label='X2')

    plt.legend(loc='upper left')
    plt.show()


def graph_average_error(satellite, prediction_data, estimate_data):
    # Graphs the average distance of measurements, x1 estimate and x2 estimates from their true values.
    average_x1_estimate_error = [0]
    average_x2_estimate_error = [0]
    average_measurements_error = [0]
    offset = 0

    for num, time in enumerate(satellite.times):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        true_x1 = satellite.x1[num]
        true_x2 = satellite.x2[num]

        if time in satellite.measurements_times:
            x1_estimate, x2_estimate, p_estimate = estimate_data[num - offset]
            x1_measurement = satellite.measurements[num - offset]
            a = average_measurements_error[-1]
            average_measurements_error.append(average(abs(x1_measurement - true_x1), num, a))
        else:
            offset += 1
            x1_estimate, x2_estimate, p_estimate = x1_prediction, x2_prediction, p_prediction

        a = average_x1_estimate_error[-1]
        average_x1_estimate_error.append(average(abs(x1_estimate - true_x1), num, a))

        a = average_x2_estimate_error[-1]
        average_x2_estimate_error.append(average(abs(x2_estimate - true_x2), num, a))

    plt.title('Average Distance of Measurements and X1, X2 Estimates')
    plt.plot(satellite.measurements_times, average_measurements_error[1:], label='Measurements')
    plt.plot(satellite.times, average_x1_estimate_error[1:], label='X1')
    plt.plot(satellite.times, average_x2_estimate_error[1:], label='X2')
    plt.legend(loc='upper left')
    plt.show()


def graph_car(car, car_data, output='123456789'):
    car_x1 = []
    car_x2 = []
    car_v1 = []
    car_v2 = []

    x1_measurements = []
    x2_measurements = []

    model_distance = []
    measurement_distance = []
    average_model_distance = [0]
    average_measurement_distance = [0]
    model_v1_distance = []
    model_v2_distance = []

    x1_p_above = []
    x1_p_below = []
    x2_p_above = []
    x2_p_below = []
    v1_p_above = []
    v1_p_below = []
    v2_p_above = []
    v2_p_below = []

    offset = 0

    for num, time in enumerate(car.times):
        state, p = car_data[num]
        x1, x2, v1, v2 = state
        car_x1.append(x1)
        car_x2.append(x2)
        car_v1.append(v1)
        car_v2.append(v2)

        true_x1 = car.x1[num]
        true_x2 = car.x2[num]
        true_v1 = car.v1[num]
        true_v2 = car.v2[num]

        distance = sqrt((true_x1-x1)**2+(true_x2-x2)**2)
        model_distance.append(distance)
        a = average_model_distance[-1]
        average_model_distance.append(average(distance,num,a))

        model_v1_distance.append(abs(true_v1 - v1))
        model_v2_distance.append(abs(true_v2 - v2))

        x1_p = p[0][0]
        x2_p = p[1][1]
        v1_p = p[2][2]
        v2_p = p[3][3]

        x1_p_above.append(x1 + three_standard_deviations(x1_p))
        x1_p_below.append(x1 - three_standard_deviations(x1_p))
        x2_p_above.append(x2 + three_standard_deviations(x2_p))
        x2_p_below.append(x2 - three_standard_deviations(x2_p))

        v1_p_above.append(v1 + three_standard_deviations(v1_p))
        v1_p_below.append(v1 - three_standard_deviations(v1_p))
        v2_p_above.append(v2 + three_standard_deviations(v2_p))
        v2_p_below.append(v2 - three_standard_deviations(v2_p))


        if time in car.measurements_times:
            x1_measurement = car.x1_measurements[num-offset]
            x1_measurements.append(x1_measurement)

            x2_measurement = car.x2_measurements[num - offset]
            x2_measurements.append(x2_measurement)

            distance = sqrt((true_x1-x1_measurement)**2+(true_x2-x2_measurement)**2)
            measurement_distance.append(distance)
            a = average_measurement_distance[-1]
            average_measurement_distance.append(average(distance, num, a))

        else:
            offset += 1

    if '0' in output:
        # Return measurement and estimate distance data
        # Used for analysis of position estimates
        return measurement_distance, model_distance

    if '1' in output:

        return model_v1_distance, model_v2_distance

    if '2' in output:
        plt.title('Path of the RC Car')
        plt.plot(car.x1, car.x2, label='True')
        #plt.plot(x1_measurements, x2_measurements,'.', label='GPS Measuements')
        plt.plot(car_x1, car_x2, label='Estimate')
        plt.legend(loc='upper left')
        plt.show()

    if '3' in output:
        plt.title('Distances of GPS Measurements and Estimates')
        plt.plot(car.measurements_times, measurement_distance, label='Measurements')
        plt.plot(car.times, model_distance, label='Estimate')
        plt.legend(loc='upper left')
        plt.show()

    if '4' in output:
        plt.title('Average Distance')
        plt.plot(car.measurements_times, average_measurement_distance[1:], label='GPS Measurements')
        plt.plot(car.times, average_model_distance[1:], label='Estimate')
        plt.legend(loc='upper left')
        plt.show()

    if '5' in output:
        plt.title('Variance of Estimate for X1')
        plt.plot(car.times, car.x1, label='True')
        plt.plot(car.times, car_x1, label='Estimate')
        plt.plot(car.times, x1_p_above, label='CI Upper Bound')
        plt.plot(car.times, x1_p_below, label='CI Lower Bound')
        plt.legend(loc='upper left')
        plt.show()

    if '6' in output:
        plt.title('Variance of Estimate for X2')
        plt.plot(car.times, car.x2, label='True')
        plt.plot(car.times, car_x2, label='Estimate')
        plt.plot(car.times, x2_p_above, label='CI Upper Bound')
        plt.plot(car.times, x2_p_below, label='CI Lower Bound')
        plt.legend(loc='upper left')
        plt.show()

    if '7' in output:
        plt.title('True and Measurements of X1')
        plt.plot(car.times, car.x1, label='True')
        plt.plot(car.measurements_times, x1_measurements,'.', label='GPS Measurements')
        plt.legend(loc='upper left')
        plt.show()

    if '8' in output:
        plt.title('True and Measurements of X2')
        plt.plot(car.times, car.x2, label='True')
        plt.plot(car.measurements_times, x2_measurements,'.', label='GPS Measurements')
        plt.legend(loc='upper left')
        plt.show()

    if 'a' in output:
        plt.title('True and Estimates of V1')
        plt.plot(car.times, car_v1, label='Estimate')
        plt.plot(car.times, v1_p_above, label='CI Upper Bound')
        plt.plot(car.times, v1_p_below, label='CI Lower Bound')
        plt.plot(car.times, car.v1, label='True')

        plt.legend(loc='upper left')
        plt.show()

    if 'b' in output:
        plt.title('True and Estimates of V2')
        plt.plot(car.times, car_v2, label='Estimate')
        plt.plot(car.times, v2_p_above, label='CI Upper Bound')
        plt.plot(car.times, v2_p_below, label='CI Lower Bound')
        plt.plot(car.times, car.v2, label='True')
        plt.legend(loc='upper left')
        plt.show()

    if '1' in output:
        plt.title('Different of V1 and V2 estimates from True values')
        plt.boxplot([model_v1_distance, model_v2_distance], showfliers=False, labels = ['V1 Estimates', 'V2 Estimates'])
