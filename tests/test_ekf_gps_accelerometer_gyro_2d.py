# tests/test_ekf_gps_accelerometer_gyro_2d.py

import unittest
from sfusion.kalman import EKFGPSAccelerometerGyro2D
import numpy as np

class TestEKFGPSAccelerometerGyro2D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0, 0, 0, 0]  # Initial positions, velocities, and orientation
        initial_covariance = np.array([[1, 0, 0, 0, 0], 
                                       [0, 1, 0, 0, 0], 
                                       [0, 0, 1, 0, 0], 
                                       [0, 0, 0, 1, 0], 
                                       [0, 0, 0, 0, 1]])  # Initial covariance
        process_noise = np.array([[1, 0, 0, 0, 0], 
                                  [0, 1, 0, 0, 0], 
                                  [0, 0, 1, 0, 0], 
                                  [0, 0, 0, 1, 0], 
                                  [0, 0, 0, 0, 1]])  # Process noise covariance
        measurement_noise = np.array([[1, 0], 
                                      [0, 1]])  # Measurement noise covariance
        
        ekf = EKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        dt = 1
        control_input = [1, 1, 0.2]  # [x_acceleration, y_acceleration, angular_velocity]
        ekf.predict(dt, control_input)
        
        # Update step
        gps_measurement = [2, 3]
        ekf.update(gps_measurement)
        
        # Validate the updated state and covariance
        expected_state = [1.61111111, 2.38888889, 1.27777778, 1.72222222, 0.31111111]
        expected_covariance = np.array([[0.76388889, -0.01388889, 0.34722222, -0.09722222, -0.11111111],
                                        [-0.01388889, 0.76388889, -0.09722222, 0.34722222, 0.11111111],
                                        [0.34722222, -0.09722222, 2.43055556, -0.68055556, -0.77777778],
                                        [-0.09722222, 0.34722222, -0.68055556, 2.43055556, 0.77777778],
                                        [-0.11111111, 0.11111111, -0.77777778, 0.77777778, 1.88888889]])

        for i in range(5):
            self.assertAlmostEqual(ekf.state[i], expected_state[i], places=4)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(ekf.covariance[i][j], expected_covariance[i][j], places=4)

if __name__ == '__main__':
    unittest.main()
