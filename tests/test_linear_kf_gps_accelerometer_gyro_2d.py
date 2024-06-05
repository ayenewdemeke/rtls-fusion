# tests/test_linear_kf_gps_accelerometer_gyro_2d.py

import unittest
from kalman_filter import LinearKFGPSAccelerometerGyro2D

class TestLinearKFGPSAccelerometerGyro2D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0, 0, 0, 0]  # Initial positions and velocities
        initial_covariance = [[1, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0], 
                              [0, 0, 1, 0, 0], 
                              [0, 0, 0, 1, 0], 
                              [0, 0, 0, 0, 1]]  # Initial covariance
        process_noise = [[1, 0, 0, 0, 0], 
                         [0, 1, 0, 0, 0], 
                         [0, 0, 1, 0, 0], 
                         [0, 0, 0, 1, 0], 
                         [0, 0, 0, 0, 1]]  # Process noise covariance
        measurement_noise = [[1, 0, 0], 
                             [0, 1, 0], 
                             [0, 0, 1]]  # Measurement noise covariance
        
        kf = LinearKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        dt = 1
        acceleration = [1, 1]
        angular_velocity = 0.2  # in radians
        kf.predict(dt, acceleration, angular_velocity)
        
        # Update step
        gps_measurement = [2, 3]
        kf.update(gps_measurement)
        
        # Validate the updated state and covariance
        expected_state = [1.625, 2.375, 1.375, 1.625, 0.59026544]
        expected_covariance = [[0.75, 0.0, 0.25, 0.0, 0.0],
                               [0.0, 0.75, 0.0, 0.25, 0.0],
                               [0.25, 0.0, 1.75, 0.0, 0.0],
                               [0.0, 0.25, 0.0, 1.75, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.66666667]]
        
        for i in range(5):
            self.assertAlmostEqual(kf.state[i], expected_state[i], places=2)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(kf.covariance[i][j], expected_covariance[i][j], places=2)

if __name__ == '__main__':
    unittest.main()
