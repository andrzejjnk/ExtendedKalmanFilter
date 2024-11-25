import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error

from ekf_plotting import plot_angles, calculate_mse

# Load the dataset
data = pd.read_csv('data/train.csv')

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

# Process Noise Covariance (Q)
kf.Q = np.diag([0.015**2, 0.015**2, 0.005**2, 0.015**2, 0.015**2, 0.005**2])

# Measurement Noise Covariance (R)
kf.R = np.diag([0.015**2, 0.015**2, 0.005**2])

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
euler_angles = []  # Store [roll, pitch, yaw]
for i in range(len(time)):
    # Predict step
    kf.predict()

    # Measurement (from accelerometer for roll/pitch, gyroscope for yaw)
    roll, pitch = acc_to_roll_pitch(accData[i])
    yaw_rate = gyroData[i, 2] * dt  # Integrate yaw rate for yaw angle
    kf.x[2] += yaw_rate  # Update yaw estimate in state vector
    measurement = np.array([roll, pitch, kf.x[2]])

    # Update step
    kf.update(measurement)

    # Store results
    euler_angles.append(kf.x[:3])

# Convert to DataFrame for saving
euler_angles = np.array(euler_angles)
data_out = pd.DataFrame({
    "Id": np.arange(1, len(euler_angles) + 1),
    "pitch": np.degrees(euler_angles[:, 1]),
    "roll": np.degrees(euler_angles[:, 0]),
    "yaw": np.degrees(euler_angles[:, 2]) / 1000
})
data_out.to_csv("orientation_test_output_filterpy.csv", index=False, float_format="%.2f")

plot_angles("orientation_test_output_filterpy.csv", "data/train.csv")
calculate_mse("data/train.csv", "orientation_test_output_filterpy.csv")
