# tests/kalman_filter/test_linear_kf_gps_1d.py

import unittest
from kalman_filter import LinearKFGPS1D

class TestLinearKFGPS1D(unittest.TestCase):
    def test_predict_update(self):
        initial_state = [0, 0]  # initial position and velocity
        initial_covariance = [[1, 0], [0, 1]]  # initial covariance
        process_noise = [[1, 0], [0, 1]]  # process noise covariance
        measurement_noise = [[1]]  # measurement noise covariance
        
        kf = LinearKFGPS1D(initial_state, initial_covariance, process_noise, measurement_noise)
        
        # Prediction step
        kf.predict(dt=1)
        self.assertAlmostEqual(kf.state[0], 0)
        self.assertAlmostEqual(kf.state[1], 0)
        
        # Update step
        kf.update(measurement=2)
        self.assertAlmostEqual(kf.state[0], 1.5)
        self.assertAlmostEqual(kf.state[1], 0.5)
        self.assertAlmostEqual(kf.covariance[0][0], 0.75)
        self.assertAlmostEqual(kf.covariance[0][1], 0.25)
        self.assertAlmostEqual(kf.covariance[1][0], 0.25)
        self.assertAlmostEqual(kf.covariance[1][1], 1.75)

if __name__ == '__main__':
    unittest.main()