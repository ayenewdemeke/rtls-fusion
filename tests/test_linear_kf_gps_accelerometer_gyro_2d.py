# tests/kalman_filter/test_linear_kf_gps_accelerometer_gyro_2d.py

import unittest
from kalman_filter import LinearKFGPSAccelerometerGyro2D

class TestLinearKFGPSAccelerometerGyro2D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0, 0, 0, 0]  # initial positions, velocities, and orientation
        initial_covariance = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]  # initial covariance
        process_noise = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]  # process noise covariance
        measurement_noise = [[1, 0], [0, 1]]  # measurement noise covariance
        
        kf = LinearKFGPSAccelerometerGyro2D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        acceleration = [1, 1]
        angular_velocity = 0.2  # in radians
        kf.predict(dt=1, acceleration=acceleration, angular_velocity=angular_velocity)
        self.assertAlmostEqual(kf.state[0], 0.5)
        self.assertAlmostEqual(kf.state[1], 0.5)
        self.assertAlmostEqual(kf.state[2], 1)
        self.assertAlmostEqual(kf.state[3], 1)
        self.assertAlmostEqual(kf.state[4], 0.2)
        
        # Update step
        kf.update(measurement=[2, 3])
        
        # Expected values based on correct computation
        expected_state = [1.625, 2.16666667, 1.375, 1.0, 0.2]
        expected_covariance = [[0.75, 0.0, 0.25, 0.0, 0.0],
                               [0.0, 0.66666667, 0.0, 0.0, 0.0],
                               [0.25, 0.0, 1.75, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 2.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 2.0]]
        
        for i in range(5):
            self.assertAlmostEqual(kf.state[i], expected_state[i])
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(kf.covariance[i][j], expected_covariance[i][j])

if __name__ == '__main__':
    unittest.main()
