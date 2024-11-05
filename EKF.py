import numpy as np

class EKF:
    """
    EKF Class for computing Extended Kalman Filter
    Based on measurements from MARG-type sensors (now only gyro and accelerometer)
    to compute orientation of an object with reference to the ENU (East, North, Up) reference frame.
    """

    def __init__(self, gyroData, accData, time, gyroNoise=None, accNoise=None):
        """
        Constructor for the EKF class.

        Parameters:
        - gyroData: Nx3 array of angular velocity [rad/s]
        - accData: Nx3 array of linear acceleration [m/s^2]
        - time: N-element array of time stamps [ms]
        - gyroNoise: 3-element array of gyroscope noise variances
        - accNoise: 3-element array of accelerometer noise variances
        """
        # Input data
        self.gyroData = gyroData
        self.accData = accData
        self.time = time
        self.num_steps = len(time)

        # Set default noise values if none are provided
        if gyroNoise is None:
            self.gyroNoise = [0.3 ** 2, 0.3 ** 2, 0.3 ** 2]
        else:
            self.gyroNoise = gyroNoise

        if accNoise is None:
            self.accNoise = [0.5 ** 2, 0.5 ** 2, 0.5 ** 2]
        else:
            self.accNoise = accNoise

        # Initial covariance matrix - make it 3D to store for each time step
        self.P = np.zeros((4, 4, self.num_steps))
        self.P[:, :, 0] = np.eye(4)  # Set the initial covariance matrix

        # Process data containers
        self.orientation = np.zeros((self.num_steps, 4))  # Store quaternion orientation [w, x, y, z]
        
        # Measurement data (normalized)
        self.measurements = None

        # Gravity reference vector
        self.g = np.array([0, 0, 1])

        # Initialize orientation and measurements
        self.normalize_measurements()
        self.initialize_orientation()

    def normalize_measurements(self):
        """
        Normalize accelerometer measurements.
        Normalization is done row-wise so that each measurement vector has unit length.
        """
        epsilon = 1e-8
        norm_acc = self.accData / (np.linalg.norm(self.accData, axis=1, keepdims=True) + epsilon)
        self.measurements = norm_acc  # Shape: (N, 3)

    def initialize_orientation(self):
        """
        Initialize the orientation quaternion based on the first set of measurements.
        This is done using an accelerometer-based initialization.
        """
        q = np.array([1, 0, 0, 0])  # Starting with a neutral orientation (identity quaternion)
        self.orientation[0] = q

    def f(self, q, w, dt):
        """
        State prediction function.

        Parameters:
        - q: Current state quaternion [w, x, y, z]
        - w: Angular velocity vector [wx, wy, wz]
        - dt: Time step

        Returns:
        - q_hat: Predicted state quaternion
        """
        qw, qx, qy, qz = q
        wx, wy, wz = w

        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -qx * wx - qy * wy - qz * wz,
             qw * wx + qy * wz - qz * wy,
             qw * wy - qx * wz + qz * wx,
             qw * wz + qx * wy - qy * wx
        ])

        # Euler integration
        q_hat = q + dt * q_dot
        q_hat /= np.linalg.norm(q_hat)  # Normalize quaternion

        return q_hat

    def F_jacobian(self, w, dt):
        """
        Compute the Jacobian of the state prediction function f with respect to the state quaternion.

        Parameters:
        - w: Angular velocity vector [wx, wy, wz]
        - dt: Time step

        Returns:
        - F: 4x4 Jacobian matrix
        """
        wx, wy, wz = w
        dq = dt / 2.0

        F = np.array([
            [1,     -dq * wx, -dq * wy, -dq * wz],
            [dq * wx,    1,  dq * wz, -dq * wy],
            [dq * wy, -dq * wz,    1,  dq * wx],
            [dq * wz,  dq * wy, -dq * wx,    1]
        ])

        return F

    def Q_process_noise(self, q, gyroNoise, dt):
        """
        Compute the process noise covariance matrix Q.

        Parameters:
        - q: Current state quaternion [w, x, y, z]
        - gyroNoise: 3-element array of gyroscope noise variances
        - dt: Time step

        Returns:
        - Q: 4x4 process noise covariance matrix
        """
        qw, qx, qy, qz = q
        wx_noise, wy_noise, wz_noise = gyroNoise

        E = np.diag([wx_noise, wy_noise, wz_noise])  # 3x3 noise covariance

        # Jacobian of the process model with respect to the noise
        W = (dt / 2.0) * np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])  # Shape: (4,3)

        Q = W @ E @ W.T  # Shape: (4,4)

        return Q

    def P_hat_prediction(self, P, q, w, gyroNoise, dt):
        """
        Predict the next state covariance matrix P_hat.

        Parameters:
        - P: Current state covariance matrix (4x4)
        - q: Current state quaternion [w, x, y, z]
        - w: Angular velocity vector [wx, wy, wz]
        - gyroNoise: 3-element array of gyroscope noise variances
        - dt: Time step

        Returns:
        - P_hat: Predicted state covariance matrix (4x4)
        """
        F = self.F_jacobian(w, dt)
        Q = self.Q_process_noise(q, gyroNoise, dt)
        P_hat = F @ P @ F.T + Q
        return P_hat

    def h(self, q, g):
        """
        Measurement function mapping the state quaternion to expected accelerometer measurements.

        Parameters:
        - q: State quaternion [w, x, y, z]
        - g: Gravity vector in ENU frame [gx, gy, gz]

        Returns:
        - h: 3-element measurement prediction vector
        """
        qw, qx, qy, qz = q
        gx, gy, gz = g

        # Accelerometer measurement prediction (gravity)
        acc_pred = 2 * np.array([
            gx * (0.5 - qy**2 - qz**2) +
            gy * (qw * qz + qx * qy) +
            gz * (qx * qz - qw * qy),

            gx * (qx * qy - qw * qz) +
            gy * (0.5 - qx**2 - qz**2) +
            gz * (qw * qx + qy * qz),

            gx * (qw * qy + qx * qz) +
            gy * (qy * qz - qw * qx) +
            gz * (0.5 - qx**2 - qy**2)
        ])

        return acc_pred

    def H_jacobian(self, q, g):
        """
        Compute the Jacobian matrix H of the measurement function h with respect to the state quaternion.

        Parameters:
        - q: State quaternion [w, x, y, z]
        - g: Gravity vector in ENU frame [gx, gy, gz]

        Returns:
        - H: 3x4 Jacobian matrix
        """
        qw, qx, qy, qz = q
        gx, gy, gz = g

        # Jacobian with respect to accelerometer prediction only
        H = np.zeros((3, 4))

        H[0, :] = 2 * np.array([
            gy * qz - gz * qy,
            gx * (-2 * qy) + gy * qw + gz * qz,
            -2 * gx * qy + gy * qz - gz * qw,
            -2 * gx * qz - gy * qy + gz * qx
        ])

        H[1, :] = 2 * np.array([
            gy * qw + gz * qx,
            gx * qy + gy * qx + gz * qz,
            gx * qx - gy * qw + gz * qz,
            gx * qz - gy * qx - gz * qw
        ])

        H[2, :] = 2 * np.array([
            -gx * qy + gy * qx,
            gx * qz + gy * qy - gz * qw,
            gx * qw - gy * qz + gz * qx,
            gx * qx + gy * qy - gz * qw
        ])

        return H

    def run(self):
        """
        Run the Extended Kalman Filter (EKF) over all time steps.
        """
        for t in range(1, self.num_steps):
            # Compute time step
            dt = (self.time[t] - self.time[t - 1]) / 1000.0  # Convert ms to seconds

            # Get current state and measurement
            q = self.orientation[t - 1]
            w = self.gyroData[t]
            acc_measured = self.measurements[t]

            # Prediction Step
            q_hat = self.f(q, w, dt)  # Predict next state
            P_hat = self.P_hat_prediction(self.P[:, :, t - 1], q, w, self.gyroNoise, dt)  # Predict covariance

            # Measurement Update Step
            acc_pred = self.h(q_hat, self.g)  # Predict accelerometer measurement
            H = self.H_jacobian(q_hat, self.g)  # Jacobian of measurement function

            # Kalman gain
            R = np.diag(self.accNoise)  # Measurement noise covariance
            S = H @ P_hat @ H.T + R  # Residual covariance
            K = P_hat @ H.T @ np.linalg.inv(S)  # Kalman gain

            # Update quaternion estimate with measurement residual
            y_tilde = acc_measured - acc_pred  # Measurement residual
            q_hat = q_hat + K @ y_tilde  # Update state estimate
            q_hat /= np.linalg.norm(q_hat)  # Re-normalize quaternion

            # Update covariance
            self.P[:, :, t] = (np.eye(4) - K @ H) @ P_hat

            # Save the orientation estimate
            self.orientation[t] = q_hat

        return self.orientation  # Return estimated quaternion orientations

