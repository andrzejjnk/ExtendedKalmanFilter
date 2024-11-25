import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error

from ekf_plotting import plot_angles, calculate_mse

# Load the dataset
data = pd.read_csv('data/test.csv')

time = data['Time'].values
gyroData = data[['GyroX', 'GyroY', 'GyroZ']].values * np.pi / 180  # Convert deg/s to rad/s
accData = data[['AccX', 'AccY', 'AccZ']].values * 9.81  # Convert g to m/s²

# Określenie liczby próbek w stanie spoczynku (potrzebne do kalibracji)
num_stationary_samples = 1000

# Krok 1: Kalibracja żyroskopu
gyro_bias = np.mean(gyroData[:num_stationary_samples], axis=0)
gyroData -= gyro_bias

# Krok 2: Kalibracja akcelerometru
acc_bias = np.mean(accData[:num_stationary_samples], axis=0) - np.array([0, 0, 9.81])
accData -= acc_bias

# Initialize Kalman Filter
# State: [roll, pitch, yaw, d_roll, d_pitch, d_yaw], Measurement: [roll, pitch, yaw]
kf = KalmanFilter(dim_x=6, dim_z=3)

# State Transition Matrix (F)
dt = np.mean(np.diff(time))  # Approximate time step
kf.F = np.array([
    [1, 0, 0, dt,  0,  0],
    [0, 1, 0,  0, dt,  0],
    [0, 0, 1,  0,  0, dt],
    [0, 0, 0,  1,  0,  0],
    [0, 0, 0,  0,  1,  0],
    [0, 0, 0,  0,  0,  1]
])

# Measurement Matrix (H)
kf.H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

# Process Noise Covariance (Q) - TUNED
kf.Q = np.diag([0.01**2, 0.01**2, 0.005**2, 0.02**2, 0.02**2, 0.005**2]) * 10

# Measurement Noise Covariance (R) - TUNED
kf.R = np.diag([0.02**2, 0.02**2, 0.005**2]) * 10

# Initial State and Covariance
kf.x = np.zeros(6)  # Initial state: [roll, pitch, yaw, 0, 0, 0]
kf.P *= 10  # Initial uncertainty

# Helper: Estimate roll, pitch from accelerometer


def acc_to_roll_pitch(acc):
    ax, ay, az = acc
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


# Run Kalman Filter
forward_states = []  # Store state vectors
forward_covs = []    # Store covariances
euler_angles = []    # Store [roll, pitch, yaw]

filterpy_data = pd.read_csv('orientation_test_output.csv')

for i in range(len(time)):
    # Predict step
    kf.predict()

    # Measurement (from accelerometer for roll/pitch, gyroscope for yaw)
    roll, pitch = acc_to_roll_pitch(accData[i])

    # # load from the other model
    roll, pitch = filterpy_data['roll'][i], filterpy_data['pitch'][i]
    roll *= np.pi / 180
    pitch *= np.pi / 180

    yaw_rate = gyroData[i, 2] * dt  # Integrate yaw rate for yaw angle
    kf.x[2] += yaw_rate  # Update yaw estimate in state vector
    measurement = np.array([roll, pitch, kf.x[2]])

    # Update step
    kf.update(measurement)

    # Store results
    forward_states.append(kf.x.copy())
    forward_covs.append(kf.P.copy())
    euler_angles.append(kf.x[:3])

# Convert to numpy arrays
forward_states = np.array(forward_states)
forward_covs = np.array(forward_covs)

# RTS Smoother
smoothed_states = np.copy(forward_states)
for t in range(len(time)-2, -1, -1):
    F = kf.F
    Q = kf.Q

    predicted_state = F @ forward_states[t]
    predicted_cov = F @ forward_covs[t] @ F.T + Q

    K = forward_covs[t] @ F.T @ np.linalg.inv(predicted_cov)

    smoothed_states[t] += K @ (smoothed_states[t+1] - predicted_state)

# Convert smoothed results to DataFrame
data_out = pd.DataFrame({
    "Id": np.arange(1, len(smoothed_states) + 1),
    "pitch": np.degrees(smoothed_states[:, 1]),
    "roll": np.degrees(smoothed_states[:, 0]),
    "yaw": np.degrees(smoothed_states[:, 2]) / 1000
})
data_out.to_csv("orientation_test_output_hybrid_smoothing.csv", index=False, float_format="%.2f")

plot_angles("orientation_test_output_hybrid.csv", "data/train.csv")
calculate_mse("data/train.csv", "orientation_test_output_hybrid.csv")
