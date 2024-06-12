# tests/test_linear_kf_gps_accelerometer_1d.py

import unittest
from sfusion.kalman import LinearKFGPSAccelerometer1D

class TestLinearKFGPSAccelerometer1D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0]  # initial position and velocity
        initial_covariance = [[1, 0], [0, 1]]  # initial covariance
        process_noise = [[1, 0], [0, 1]]  # process noise covariance
        measurement_noise = [[1]]  # measurement noise covariance
        
        kf = LinearKFGPSAccelerometer1D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        kf.predict(dt=1, acceleration=1)
        self.assertAlmostEqual(kf.state[0], 0.5)
        self.assertAlmostEqual(kf.state[1], 1)
        
        # Update step
        kf.update(measurement=2)
        
        # Expected values based on correct computation
        expected_state = [1.625, 1.375]
        expected_covariance = [[0.75, 0.25],
                               [0.25, 1.75]]
        
        self.assertAlmostEqual(kf.state[0], expected_state[0])
        self.assertAlmostEqual(kf.state[1], expected_state[1])
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(kf.covariance[i][j], expected_covariance[i][j])

if __name__ == '__main__':
    unittest.main()
