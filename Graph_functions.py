import matplotlib.pyplot as plt
from helper_functions import *


def average_1(x, n, a):
    # computes an average from last average a and n indexed at 0 for next item x
    return (n * a + x) / (n + 1)


def graph_x1(satellite, prediction_data, estimate_data):
    estimate_x1 = []
    predictions_x1 = []

    for num in range(len(prediction_data)):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]

        predictions_x1.append(x1_prediction)

    for num in range(len(estimate_data)):
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]

        estimate_x1.append(x1_estimate)

    plt.title('Dimension X1: Measurements, Prediction and Estimate of X1 over time')
    plt.plot(satellite.times, satellite.x1, label='X1')
    plt.plot(satellite.measurements_times, estimate_x1, label='Estimate_x1')
    plt.plot(satellite.measurements_times, satellite.measurements, 'o', label='Measurements')
    plt.plot(satellite.times, predictions_x1, '+', label='Predictions_x1')
    plt.legend()
    plt.show()


def graph_x1_and_p(satellite, prediction_data, estimate_data):
    estimate_x1 = []
    predictions_x1 = []
    p_above = []
    p_below = []

    for num in range(len(prediction_data)):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        predictions_x1.append(x1_prediction)
        p = p_prediction[0][0]
        p_above.append(x1_prediction + two_standard_deviations(p))
        p_below.append(x1_prediction - two_standard_deviations(p))

    for num in range(len(estimate_data)):
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]
        estimate_x1.append(x1_estimate)

    plt.title('Dimension X1: Measurements, Prediction and P of X1 over time')
    plt.plot(satellite.times, satellite.x1, '.', label='X1')
    plt.plot(satellite.times, predictions_x1, label='Prediction_x1')
    plt.plot(satellite.times, p_above, label='Two Standard deviations above')
    plt.plot(satellite.times, p_below, label='Two Standard deviations below')

    plt.legend()
    plt.show()


def graph_x2(satellite, prediction_data, estimate_data):
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
    plt.plot(satellite.times, predictions_x2, '+', label='Predictions_x2')
    plt.legend()
    plt.show()


def graph_x2_and_p(satellite, prediction_data, estimate_data):
    estimate_x2 = []
    predictions_x2 = []
    p_above = []
    p_below = []

    for num in range(len(prediction_data)):
        x1_prediction, x2_prediction, p_prediction = prediction_data[num]
        p = p_prediction[1][1]
        p_above.append(x2_prediction + two_standard_deviations(p))
        p_below.append(x2_prediction - two_standard_deviations(p))
        predictions_x2.append(x2_prediction)

    for num in range(len(estimate_data)):
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]
        estimate_x2.append(x2_estimate)

    plt.title('Dimension X2: Measurements, Prediction and P of X2 over time')
    plt.plot(satellite.times, satellite.x2, '.', label='X1')
    plt.plot(satellite.times, predictions_x2, label='Prediction_x2')
    plt.plot(satellite.times, p_above, label='Two Standard deviations above')
    plt.plot(satellite.times, p_below, label='Two Standard deviations below')

    plt.legend()
    plt.show()


def graph_error(satellite, prediction_data, estimate_data):
    measurements_error = []
    x1_estimate_error = []
    x2_estimate_error = []
    h = satellite.times[1] - satellite.times[0]

    for num, time in enumerate(satellite.measurements_times):
        time = int(time // h)
        x1_prediction, x2_prediction, p_prediction = prediction_data[time]
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]
        true_x1 = satellite.x1[time]
        true_x2 = satellite.x2[time]
        x1_measurement = satellite.measurements[num]

        measurements_error.append(x1_measurement - true_x1)
        x1_estimate_error.append(x1_estimate - true_x1)
        x2_estimate_error.append(x2_estimate - true_x2)

    plt.title('Error of Measurements and X1, X2 Estimates')
    plt.plot(satellite.measurements_times, measurements_error, label='Measurements_error')
    # plt.plot(satellite.measurements_times, x1_estimate_error, label='X1_estimate_error')
    # plt.plot(satellite.measurements_times, x2_estimate_error, label='X2_estimate_error')
    plt.legend()
    plt.show()


def graph_average_error(satellite, prediction_data, estimate_data):
    average_measurements_error = [0]
    average_x1_estimate_error = [0]
    average_x2_estimate_error = [0]
    h = satellite.times[1] - satellite.times[0]

    for num, time in enumerate(satellite.measurements_times):
        time = int(time // h)
        x1_prediction, x2_prediction, p_prediction = prediction_data[time]
        x1_estimate, x2_estimate, p_estimate = estimate_data[num]
        true_x1 = satellite.x1[time]
        true_x2 = satellite.x2[time]
        x1_measurement = satellite.measurements[num]

        a = average_measurements_error[-1]
        average_measurements_error.append(average_1(abs(x1_measurement - true_x1), num, a))

        # a = average_x1_estimate_error[-1]
        # average_x1_estimate_error.append(average_1(abs(x1_estimate - true_x1), num, a))

        # a = average_x2_estimate_error[-1]
        # average_x2_estimate_error.append(average_1(abs(x2_estimate - true_x2), num, a))

    plt.title('Average Error of Measurements and X1, X2 Estimates')
    plt.plot(satellite.measurements_times, average_measurements_error[1:], label='Average_measurements_error')
    plt.plot(satellite.measurements_times, average_x1_estimate_error[1:], label='Average_x1_estimate_error')
    plt.plot(satellite.measurements_times, average_x2_estimate_error[1:], label='Average_x2_estimate_error')
    plt.legend()
    plt.show()
