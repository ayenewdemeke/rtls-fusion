# tests/test_ukf_gps_accelerometer_gyro_2d.py

import unittest
from sfusion.kalman import UKFGPSAccelerometerGyro2D
import numpy as np

class TestUKFGPSAccelerometerGyro2D(unittest.TestCase):
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
        
        ukf = UKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        dt = 1
        control_input = [1, 1, 0.2]  # [x_acceleration, y_acceleration, angular_velocity]
        ukf.predict(dt, control_input)
        
        # Update step
        gps_measurement = [2, 3]
        ukf.update(gps_measurement)
        
        # Validate the updated state and covariance
        expected_state = [1.63215064, 2.45684303, 0.87674898, 1.25985469, 0.25184993]
        expected_covariance = np.array([[-0.19166971,  0.28167542,  0.3817785,   0.03903065, -0.22427963],
                                        [ 0.28167542,  0.28167542,  0.43008963,  0.23430914,  0.23430914],
                                        [ 0.3817785,   0.43008963,  3.00202749,  0.89127792, -0.87902538],
                                        [ 0.03903065,  0.23430914, -0.30559333,  3.00202749,  0.89127792],
                                        [-0.22427963,  0.23430914, -0.87902538,  0.89127792,  1.96296299]])

        for i in range(5):
            self.assertAlmostEqual(ukf.state[i], expected_state[i], places=4)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(ukf.covariance[i][j], expected_covariance[i][j], places=4)

if __name__ == '__main__':
    unittest.main()
