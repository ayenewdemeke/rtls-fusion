# examples/kalman_filter/linear_kf_gps_accelerometer_gyro_2d.py

from kalman_filter import LinearKFGPSAccelerometerGyro2D

# Initial state [x_position, y_position, x_velocity, y_velocity, orientation]
initial_state = [0, 0, 0, 0, 0]

# Initial covariance matrix
initial_covariance = [[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]]

# Process noise covariance matrix
process_noise = [[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]

# Measurement noise covariance matrix
measurement_noise = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]

# Create the filter
kf = LinearKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)

# Predict and update
acceleration = [1, 1]
angular_velocity = 0.2  # in radians
kf.predict(dt=1, acceleration=acceleration, angular_velocity=angular_velocity)
kf.update(measurement=[2, 3])

print("State after update:", kf.state)
print("Covariance after update:", kf.covariance)
