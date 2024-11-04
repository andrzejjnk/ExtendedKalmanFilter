import numpy as np
import pandas as pd
from EKF import *

# Load data from train.csv
data = pd.read_csv('data/test.csv')

# Extract relevant columns and process them for the EKF
time = data['Time'].values  # Time in seconds (assuming it's in seconds; convert from ms if needed)
gyroData = data[['GyroX', 'GyroY', 'GyroZ']].values * np.pi / 180  # Convert degrees/sec to rad/sec if necessary
accData = data[['AccX', 'AccY', 'AccZ']].values * 9.81  # Accelerometer data
magData = data[['MagX', 'MagY', 'MagZ']].values * 100  # Magnetometer data

# Step 1: Calculate gyroscope bias
# Assuming the device was stationary at the start (adjust the index for your dataset)
# For example, you might want to consider the first 100 samples as stationary data
num_stationary_samples = 400
gyro_bias = np.mean(gyroData[:num_stationary_samples], axis=0)

# Step 2: Subtract bias from gyro data
gyroData_calibrated = gyroData - gyro_bias

# Assuming EKF is your class for the Extended Kalman Filter implementation
# Initialize the EKF with calibrated sensor data
ekf = EKF(gyroData=gyroData_calibrated, accData=accData, magData=magData, time=time)

# Run the EKF
ekf.run()

# Access the results
# Assuming `orientation` stores quaternion orientations at each step
orientations = ekf.orientation

# Print the results (e.g., first few orientation estimates)
quaternions = []
for i, orientation in enumerate(orientations):
    # print(f"Step {i+1} - Quaternion Orientation: {orientation}")
    quaternions.append(orientation)


def quaternion_to_euler(q):
    w, x, y, z = q
    
    # Calculate roll, pitch, yaw
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Convert radians to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


output = []
for i, q in enumerate(quaternions):
    roll, pitch, yaw = quaternion_to_euler(q)
    output.append(f"{i+1},{pitch:.2f},{roll:.2f},{yaw:.2f}")

# Print the output in CSV format
print("Id,pitch,roll,yaw")
for line in output:
    print(line)
