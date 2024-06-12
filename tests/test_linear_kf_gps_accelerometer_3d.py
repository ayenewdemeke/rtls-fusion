# tests/test_linear_kf_gps_accelerometer_3d.py

import unittest
from sfusion.kalman import LinearKFGPSAccelerometer3D

class TestLinearKFGPSAccelerometer3D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0, 0, 0, 0, 0]  # initial positions and velocities
        initial_covariance = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]  # initial covariance
        process_noise = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]  # process noise covariance
        measurement_noise = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # measurement noise covariance
        
        kf = LinearKFGPSAccelerometer3D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        kf.predict(dt=1, acceleration=[1, 1, 1])
        self.assertAlmostEqual(kf.state[0], 0.5)
        self.assertAlmostEqual(kf.state[1], 0.5)
        self.assertAlmostEqual(kf.state[2], 0.5)
        self.assertAlmostEqual(kf.state[3], 1)
        self.assertAlmostEqual(kf.state[4], 1)
        self.assertAlmostEqual(kf.state[5], 1)
        
        # Update step
        kf.update(measurement=[2, 3, 4])
        
        # Expected values based on correct computation
        expected_state = [1.625, 2.375, 3.125, 1.375, 1.625, 1.875]
        expected_covariance = [[0.75, 0.0, 0.0, 0.25, 0.0, 0.0],
                               [0.0, 0.75, 0.0, 0.0, 0.25, 0.0],
                               [0.0, 0.0, 0.75, 0.0, 0.0, 0.25],
                               [0.25, 0.0, 0.0, 1.75, 0.0, 0.0],
                               [0.0, 0.25, 0.0, 0.0, 1.75, 0.0],
                               [0.0, 0.0, 0.25, 0.0, 0.0, 1.75]]
        
        for i in range(6):
            self.assertAlmostEqual(kf.state[i], expected_state[i])
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(kf.covariance[i][j], expected_covariance[i][j])

if __name__ == '__main__':
    unittest.main()
