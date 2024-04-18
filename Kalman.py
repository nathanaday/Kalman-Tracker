import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


class Kalman:
    def __init__(self, process_noise_var=2, measurement_noise=64, initial_uncertainty=2500):
        """
        Initializes an instance of a Kalman filter designed for object tracking.
        Noise and measurement uncertainty parameters can be tuned based on application requirements.

        Parameters:
            - process_noise_var (float): Variance of the process noise, indicating how much we expect the
            model's predictions to deviate from the actual process. Higher values indicate less confidence in the
            model's predictions.
            - measurement_noise (float): Variance of the measurement noise, representing the expected noise
            in the measurement system. Higher values indicate less confidence in the accuracy of the measurements.
            - initial_uncertainty (float): Initial value for the diagonal elements of the state covariance matrix P.
            This value sets the initial state of uncertainty for the position and velocity estimates. Higher values
            suggest less confidence in the initial estimates.

        Attributes:
            - kf (KalmanFilter): An instance of the filterpy library's KalmanFilter class
              state and 2-dimensional measurement.
            - process_noise_var (float): Storage of the process noise variance parameter.
            - measurement_noise (float): Storage of the measurement noise variance parameter.
            - initial_uncertainty (float): Initial uncertainty used in the state covariance matrix.
            - dt (float, None): Time step in seconds between state updates; usually updating when fetching predictions
            - is_initialized (bool): Flag to indicate whether the Kalman filter has been initialized
        """

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.process_noise_var = process_noise_var
        self.measurement_noise = measurement_noise
        self.initial_uncertainty = initial_uncertainty
        self.dt = None

        self.is_initialized = False

    def initialize_filter(self, x_initial, y_initial, initial_dt):
        """ Initialized the filter with the first observation """
        self.kf.x = np.array([x_initial, y_initial, 0, 0])
        self.dt = initial_dt

        self.initialize_matrices()
        self.kf.P *= self.initial_uncertainty
        self.kf.R = self.measurement_noise
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.process_noise_var)

        self.is_initialized = True

    def initialize_matrices(self):
        """ Allows the state transition matrix to be updated with a new delta-t for applications with non-constant
        prediction timing (such as camera framerate) """
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def predict(self, dt=None):
        """ Predict the next state. Adjust dt if necessary. """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.initialize_matrices()  # Update F matrix with new dt
        self.kf.predict()
        prediction = self.kf.x[:2]
        return prediction[0], prediction[1]

    def update(self, new_x, new_y):
        """ Update the state by a new measurement. """
        z = np.array([new_x, new_y])
        self.kf.update(z)
        return self.kf.x[:2]


if __name__ == '__main__':

    # # # Example usage:
    kf = Kalman()

    # Starting at (3, 3)
    kf.initialize_filter(3, 3, 0.1)
    print(kf.predict())

    # New sensor update
    kf.update(3.1, 3.2)
    print(kf.predict())
    print(kf.predict())
    print(kf.predict())
