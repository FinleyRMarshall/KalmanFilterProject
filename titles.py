s_figure_txt = [
    'Figure 1: The satellite\'s path. Notice the path is not perfect because there is random movement.',
    'Figure 2: The measurements being observed by the satellite. We observe that the measurements are spread around the true $X_1$ values.',
    'Figure 3: The model\'s estimates for $X_1$ and the certainty of the estimates. We can see the true values for $X_1$ lies inside the CI nearly all the time.',
    'Figure 4: The model\'s estimates for $X_2$ and the certainty of the estimates. We can see the true values for $X_2$ lies inside the CI nearly all the time.',
    'Figure 5: The average distance of the estimates and measurements. We distinguish that the model is more accurate and thus performing better than the measurements.',
    'Figure 6: The performance of the model with a larger time step. We see the model still performs better than the measurements.',
    'Figure 7: The performance of the model with a larger measurement variance. We see the model still performs better than the measurements.',
    'Figure 8: The performance of the model with a larger movement variance. We see the model still performs better than the measurements.',
]

s_plt_titles = [
    'Analysis of Position Estimates',
    'Analysis of Position Estimates with an Increased Time Step',
    'Analysis of Position Estimates with an Increased Measurement Variance',
    'Analysis of Position Estimates with an Increased Movement Variance'
]

c_figure_txt = [
    'Figure 9: The RC car\'s path around a realistic racetrack. We use this exact path throughout the report.',
    'Figure 10: Compare the estimates using noisy control measurements to the true path of the RC car. Notice how quickly the two paths diverge. Our EKF model will be supplied with these noisy control measurements.',
    'Figure 11: The GPS measurements being observed when the RC car travels in a straight line. We can see how noisy these GPS measurements are. Our EKF model will be supplied with these noisy GPS measurements.',
    'Figure 12: View the model\'s estimates when the RC car travels in a straight line.',
    'Figure 13: View the model\'s estimates when the RC car travels around a corner.',
    'Figure 14: View the model\'s estimates when the RC car travels around the racetrack.',
    'Figure 15: The model\'s position estimates for $X_1$ and the certainty of the estimates. The RC car is traveling around the racetrack. We can see the true values for $X_1$ lies inside the CI nearly all the time.',
    'Figure 16: The model\'s position estimates for $X_2$ and the certainty of the estimates. The RC car is traveling around the racetrack. We can see the true values for $X_2$ lies inside the CI nearly all the time.',
    'Figure 17: The model\'s velocity estimates for $V_1$ and the certainty of the estimates. The RC car is traveling around the racetrack. We observe the true values for $V_1$ lies inside the CI nearly all the time.',
    'Figure 18: The model\'s velocity estimates for $V_2$ and the certainty of the estimates. The RC car is traveling around the racetrack. We observe the true values for $V_2$ lies inside the CI nearly all the time.',
    'Figure 19: A box plot of the position estimates distances and GPS measurements distances. We distinguish the model\'s estimates are significantly more accurate than the measurements.',
    'Figure 20: A box plot of the velocity estimates distances. Notice the box plots for $V_1$ and $V_2$ are distributed the same and there are no measurements to compare too.',
]

c_plt_titles = [
    'Analysis of Position Estimates',
    'Analysis of Velocity Estimates'
]