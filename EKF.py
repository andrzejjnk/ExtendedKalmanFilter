import numpy as np

class EKF:
    """
    EKF Class for computing Extended Kalman Filter
    Based on measurements from MARG-type sensors to compute orientation of
    an object with reference to the ENU (East, North, Up) reference frame.
    """

    def __init__(self, gyroData, accData, magData, time, gyroNoise=None, accNoise=None, magNoise=None):
        """
        Constructor for the EKF class.

        Parameters:
        - gyroData: Nx3 array of angular velocity [rad/s]
        - accData: Nx3 array of linear acceleration [m/s^2]
        - magData: Nx3 array of magnetic field strength [uT]
        - time: N-element array of time stamps [ms]
        - accNoise: 3-element array of accelerometer noise variances
        - magNoise: 3-element array of magnetometer noise variances
        """
        # Input data
        self.gyroData = gyroData
        self.accData = accData
        self.magData = magData
        self.time = time
        self.num_steps = len(time)

        # Set default noise values if none are provided
        if gyroNoise is None:
            self.gyroNoise = [0.02 ** 2, 0.02 ** 2, 0.02 ** 2] # Example: [variance_x, variance_y, variance_z]
        else:
            self.gyroNoise = gyroNoise

        if accNoise is None:
            self.accNoise = [0.02 ** 2, 0.02 ** 2, 0.02 ** 2]  # Example: [variance_x, variance_y, variance_z]
        else:
            self.accNoise = accNoise

        if magNoise is None:
            self.magNoise = [0.02 ** 2, 0.02 ** 2, 0.02 ** 2]  # Example: [variance_x, variance_y, variance_z]
        else:
            self.magNoise = magNoise

        # Initial covariance matrix - make it 3D to store for each time step
        self.P = np.zeros((4, 4, self.num_steps))
        self.P[:, :, 0] = np.eye(4)  # Set the initial covariance matrix

        # Process data containers
        self.orientation = np.zeros((self.num_steps, 4))  # Store quaternion orientation [w, x, y, z]
        
        # Measurement data (normalized)
        self.measurements = None

        # Constants
        self.g = np.array([0, 0, 1])
        self.r = np.array([0, np.cos(np.deg2rad(66)), -np.sin(np.deg2rad(66))])

        # Initialize orientation and measurements
        self.normalize_measurements()
        self.initialize_orientation()

    def normalize_measurements(self):
        """
        Normalize accelerometer and magnetometer measurements.
        Normalization is done row-wise so that each measurement vector has unit length.
        """
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        norm_acc = self.accData / (np.linalg.norm(self.accData, axis=1, keepdims=True) + epsilon)
        norm_mag = self.magData / (np.linalg.norm(self.magData, axis=1, keepdims=True) + epsilon)
        self.measurements = np.hstack((norm_acc, norm_mag))  # Shape: (N, 6)

    def initialize_orientation(self):
        """
        Initialize the orientation quaternion based on the first set of measurements.
        This is done using an e-compass method.
        """
        q = np.array([1, 0, 0, 0])  # Starting with a neutral orientation (identity quaternion)
        self.orientation[0] = q  # Assign the initial quaternion
        a_measure = self.measurements[0, 0:3]  # First accelerometer measurement
        m_measure = self.measurements[0, 3:6]  # First magnetometer measurement
        m_ref = self.r
        a_ref = self.g

        # Compute normalized cross products
        cross_ref = np.cross(m_ref, a_ref)
        norm_cross_ref = np.linalg.norm(cross_ref) + 1e-8
        n_cross_ref = cross_ref / norm_cross_ref

        cross_meas = np.cross(m_measure, a_measure)
        norm_cross_meas = np.linalg.norm(cross_meas) + 1e-8
        n_cross_meas = cross_meas / norm_cross_meas

        # Construct matrix A
        A_ref = np.column_stack((m_ref, n_cross_ref, np.cross(m_ref, n_cross_ref)))
        A_meas = np.column_stack((m_measure, n_cross_meas, np.cross(m_measure, n_cross_meas)))

        # Compute A matrix product
        A = A_ref @ A_meas.T

        # Compute quaternion components
        q0 = 0.5 * np.sqrt(1 + A[0, 0] + A[1, 1] + A[2, 2])
        if q0 < 1e-8:
            q0 = 1e-8  # Prevent division by zero

        q1 = (A[2, 1] - A[1, 2]) / (4 * q0)
        q2 = (A[0, 2] - A[2, 0]) / (4 * q0)
        q3 = (A[1, 0] - A[0, 1]) / (4 * q0)

        # Normalize the quaternion
        q = np.array([q0, q1, q2, q3])
        q /= np.linalg.norm(q)

        # Assign the computed quaternion for the initial orientation
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

    def h(self, q, g, r):
        """
        Measurement function mapping the state quaternion to expected sensor measurements.

        Parameters:
        - q: State quaternion [w, x, y, z]
        - g: Gravity vector in ENU frame [gx, gy, gz]
        - r: Magnetic field vector in ENU frame [rx, ry, rz]

        Returns:
        - h: 6-element measurement prediction vector
        """
        qw, qx, qy, qz = q
        gx, gy, gz = g
        rx, ry, rz = r

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

        # Magnetometer measurement prediction (magnetic field)
        mag_pred = 2 * np.array([
            rx * (0.5 - qy**2 - qz**2) +
            ry * (qw * qz + qx * qy) +
            rz * (qx * qz - qw * qy),

            rx * (qx * qy - qw * qz) +
            ry * (0.5 - qx**2 - qz**2) +
            rz * (qw * qx + qy * qz),

            rx * (qw * qy + qx * qz) +
            ry * (qy * qz - qw * qx) +
            rz * (0.5 - qx**2 - qy**2)
        ])

        h = np.hstack((acc_pred, mag_pred))  # Shape: (6,)

        return h

    def H_jacobian(self, q, g, r):
        """
        Compute the Jacobian matrix H of the measurement function h with respect to the state quaternion.

        Parameters:
        - q: State quaternion [w, x, y, z]
        - g: Gravity vector in ENU frame [gx, gy, gz]
        - r: Magnetic field vector in ENU frame [rx, ry, rz]

        Returns:
        - H: 6x4 Jacobian matrix
        """
        qw, qx, qy, qz = q
        gx, gy, gz = g
        rx, ry, rz = r

        # Precompute terms
        H = np.zeros((6, 4))

        # Accelerometer Jacobian
        H[0, :] = 2 * np.array([
            gy * qz - gz * qy,
            gx * ( -2 * qy ) + gy * qw + gz * qz,
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
            gy * qz - gz * qy,
            gx * qz + gy * qw - gz * qx,
            gx * qx + gy * qz - gz * qy,
            gx * qy + gy * qx - gz * qx
        ])

        # Magnetometer Jacobian
        H[3, :] = 2 * np.array([
            ry * qz - rz * qy,
            rx * ( -2 * qy ) + ry * qw + rz * qz,
            -2 * rx * qy + ry * qz - rz * qw,
            -2 * rx * qz - ry * qy + rz * qx
        ])

        H[4, :] = 2 * np.array([
            ry * qw + rz * qx,
            rx * qy + ry * qx + rz * qz,
            rx * qx - ry * qw + rz * qz,
            rx * qz - ry * qx - rz * qw
        ])

        H[5, :] = 2 * np.array([
            ry * qz - rz * qy,
            rx * qz + ry * qw - rz * qx,
            rx * qx + ry * qz - rz * qy,
            rx * qy + ry * qx - rz * qx
        ])

        return H  # Shape: (6,4)

    def R_measurement_noise(self):
        """
        Compute the measurement noise covariance matrix R.

        Returns:
        - R: 6x6 measurement noise covariance matrix
        """
        diag = np.hstack((self.accNoise, self.magNoise))
        R = np.diag(diag)
        return R

    def innovation(self, q_hat, z):
        """
        Compute the innovation vector.

        Parameters:
        - q_hat: Predicted state quaternion [w, x, y, z]
        - z: Current measurement vector (6,)

        Returns:
        - v: Innovation vector (6,)
        """
        h_pred = self.h(q_hat, self.g, self.r)
        v = z - h_pred
        return v

    def S_matrix(self, H, P_hat, R):
        """
        Compute the innovation covariance matrix S.

        Parameters:
        - H: Measurement Jacobian matrix (6x4)
        - P_hat: Predicted state covariance matrix (4x4)
        - R: Measurement noise covariance matrix (6x6)

        Returns:
        - S: Innovation covariance matrix (6x6)
        """
        S = H @ P_hat @ H.T + R
        return S

    def Kalman_gain(self, P_hat, H, S):
        """
        Compute the Kalman Gain matrix K.

        Parameters:
        - P_hat: Predicted state covariance matrix (4x4)
        - H: Measurement Jacobian matrix (6x4)
        - S: Innovation covariance matrix (6x6)

        Returns:
        - K: Kalman Gain matrix (4x6)
        """
        K = P_hat @ H.T @ np.linalg.inv(S)
        return K

    def update_P(self, K, H, P_hat):
        """
        Update the state covariance matrix P after measurement update.

        Parameters:
        - K: Kalman Gain matrix (4x6)
        - H: Measurement Jacobian matrix (6x4)
        - P_hat: Predicted state covariance matrix (4x4)

        Returns:
        - P_new: Updated state covariance matrix (4x4)
        """
        I = np.eye(4)
        P_new = (I - K @ H) @ P_hat
        return P_new

    def run(self):
        """
        Execute the main EKF loop to estimate orientation over time.
        """
        N = self.gyroData.shape[0]
        self.orientation[0] = self.orientation[0]  # Set initial orientation (this line is unnecessary)

        # Initialize P for each time step
        # self.P is already correctly sized in the constructor

        R = self.R_measurement_noise()

        for i in range(1, N):
            # Time difference
            dt = self.time[i] - self.time[i - 1]  # Assuming time is in milliseconds
            dt /= 1000.0  # Convert to seconds if needed

            # Previous state
            q_prev = self.orientation[i - 1]

            # Angular velocity
            w = self.gyroData[i]

            # Prediction step
            q_hat = self.f(q_prev, w, dt)
            P_hat = self.P[:, :, i - 1]
            P_hat = self.P_hat_prediction(P_hat, q_prev, w, self.gyroNoise, dt)

            # Measurement
            z = self.measurements[i]

            # Innovation
            v = self.innovation(q_hat, z)

            # Measurement Jacobian
            H = self.H_jacobian(q_hat, self.g, self.r)

            # Innovation covariance
            S = self.S_matrix(H, P_hat, R)

            # Kalman Gain
            K = self.Kalman_gain(P_hat, H, S)

            # State update
            q_new = q_hat + K @ v

            # Normalize quaternion
            q_new /= np.linalg.norm(q_new)

            self.orientation[i] = q_new  # Assign the updated quaternion

            # Covariance update
            P_new = self.update_P(K, H, P_hat)
            self.P[:, :, i] = P_new  # Update the covariance matrix

    def get_orientation(self):
        """
        Retrieve the estimated orientation over time.

        Returns:
        - orientation: Nx4 array of quaternions [w, x, y, z]
        """
        return self.orientation

    def get_covariance(self):
        """
        Retrieve the state covariance matrices over time.

        Returns:
        - P: 4x4xN array of covariance matrices
        """
        return self.P
