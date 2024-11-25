from ekf_plotting import plot_angles, calculate_mse
import numpy as np
import pandas as pd
import csv
import math
from EKF import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data/test.csv')

time = data['Time'].values  # Zakładamy, że czas jest w sekundach
gyroData = data[['GyroX', 'GyroY', 'GyroZ']].values * \
    np.pi / 180  # Zmiana jednostek z deg/s na rad/s
accData = data[['AccX', 'AccY', 'AccZ']].values * 9.81  # Zmiana jednostek na m/s²
# Zmiana jednostek magnetometru (jeśli dane są w mG) (było razy 1000) # 15.11.2024 Zakładamy, że dane są w mikro Teslach, a chcemy uzyskać dane w Gaussach więc dzielimy przez 100
magData = data[['MagX', 'MagY', 'MagZ']].values / 100

# Określenie liczby próbek w stanie spoczynku (potrzebne do kalibracji)
num_stationary_samples = 1000

# Krok 1: Kalibracja żyroskopu
gyro_bias = np.mean(gyroData[:num_stationary_samples], axis=0)
gyroData_calibrated = gyroData - gyro_bias

# Krok 2: Kalibracja akcelerometru
acc_bias = np.mean(accData[:num_stationary_samples], axis=0) - np.array([0, 0, 9.81])
accData_calibrated = accData - acc_bias

ekf = EKF(gyroData=gyroData_calibrated, accData=accData_calibrated, time=time)

# główna funkcja EKFa
ekf.run()

# ekf.orientation stores quaternion orientations at each step
orientations = ekf.orientation

quaternions = []
for i, orientation in enumerate(orientations):
    quaternions.append(orientation)

# Source: Quaternion to Euler angles (in 3-2-1 sequence)
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles


def quaternion_to_euler(q):
    w, x, y, z = q

    # Calculate roll, pitch, yaw
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convert radians to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# Unwrap yaw angles


def unwrap_yaw(yaw_angles):
    """
    Unwrap the yaw angles to remove discontinuities.
    """
    return np.unwrap(np.radians(yaw_angles)) * (180 / np.pi)


# Yaw is better here
filterpy_data = pd.read_csv('orientation_test_output_hybrid_smoothing.csv')

# Compute roll, pitch, yaw and unwrap yaw
euler_angles = []
yaw_angles = []
for i, q in enumerate(quaternions):
    roll, pitch, yaw = quaternion_to_euler(q)

    # Overwrite yaw
    yaw = filterpy_data['yaw'][i]

    euler_angles.append((roll, pitch, yaw))
    yaw_angles.append(yaw)

# Unwrap the yaw angles
yaw_angles_unwrapped = unwrap_yaw(yaw_angles)

# Generate the output data
output = []
for i, (roll, pitch, yaw) in enumerate(euler_angles):
    yaw_unwrapped = yaw_angles_unwrapped[i]
    output.append(f"{i+1},{pitch:.2f},{roll:.2f},{yaw_unwrapped:.2f}")

# Print the output in CSV format
# print("Id,pitch,roll,yaw")
# for line in output:
#     print(line)


data = {
    "Id": [i + 1 for i in range(len(quaternions))],
    "pitch": [euler[1] for euler in euler_angles],
    "roll": [euler[0] for euler in euler_angles],
    "yaw": yaw_angles_unwrapped,  # Use the unwrapped yaw angles
}

df = pd.DataFrame(data)
df.to_csv("orientation_test_output.csv", index=False, float_format="%.4f")


plot_angles("orientation_test_output.csv", 'data/train.csv')
calculate_mse("data/train.csv", "orientation_test_output.csv")
