# examples/kalman_filter/linear_kf_gps_accelerometer_2d.py

from kalman_filter import LinearKFGPSAccelerometer2D

# Initial state [x_position, y_position, x_velocity, y_velocity]
initial_state = [0, 0, 0, 0]

# Initial covariance matrix
initial_covariance = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

# Process noise covariance matrix
process_noise = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

# Measurement noise covariance matrix
measurement_noise = [[1, 0],
                     [0, 1]]

# Create the filter
kf = LinearKFGPSAccelerometer2D(initial_state, initial_covariance, process_noise, measurement_noise)

# Predict and update
kf.predict(dt=1, acceleration=[1, 1])
kf.update(measurement=[2, 3])

print("State after update:", kf.state)
print("Covariance after update:", kf.covariance)
