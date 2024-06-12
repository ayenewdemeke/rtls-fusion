# tests/test_linear_kf_gps_accelerometer_gyro_2d.py

import unittest
from sfusion.kalman import LinearKFGPSAccelerometerGyro2D

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
        measurement_noise = [[1, 0], 
                             [0, 1]]  # Measurement noise covariance
        
        kf = LinearKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        dt = 1
        control_input = [1, 1, 0.2]  # [x_acceleration, y_acceleration, angular_velocity]
        kf.predict(dt, control_input)
        
        # Update step
        gps_measurement = [2, 3]
        kf.update(gps_measurement)
        
        # Validate the updated state and covariance
        expected_state = [1.625, 2.375, 1.375, 1.625, 0.2]
        expected_covariance = [[0.75, 0.0, 0.25, 0.0, 0.0],
                               [0.0, 0.75, 0.0, 0.25, 0.0],
                               [0.25, 0.0, 1.75, 0.0, 0.0],
                               [0.0, 0.25, 0.0, 1.75, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 2.0]]
        
        for i in range(5):
            self.assertAlmostEqual(kf.state[i], expected_state[i])
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(kf.covariance[i][j], expected_covariance[i][j])

if __name__ == '__main__':
    unittest.main()
