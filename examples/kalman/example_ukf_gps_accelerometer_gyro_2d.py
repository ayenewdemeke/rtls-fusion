# examples/kalman/ukf_gps_accelerometer_gyro_2d_example.py

from sfusion.kalman import UKFGPSAccelerometerGyro2D
import numpy as np

# Initial state [x_position, y_position, x_velocity, y_velocity, orientation]
initial_state = [0, 0, 0, 0, 0]

# Initial covariance matrix
initial_covariance = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]])

# Process noise covariance matrix
process_noise = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])

# Measurement noise covariance matrix
measurement_noise = np.array([[1, 0],
                              [0, 1]])

# Create the filter
ukf = UKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)

# Predict and update
control_input = [1, 1, 0.2]  # [x_acceleration, y_acceleration, angular_velocity]
ukf.predict(dt=1, control_input=control_input)
ukf.update(measurement=[2, 3])

print("State after update:", ukf.state)
print("Covariance after update:", ukf.covariance)
