import numpy as np
import pandas as pd
import csv
import math
from EKF import *
import matplotlib.pyplot as plt

# Load data from train.csv
data = pd.read_csv('data/train.csv')

# Extract relevant columns and process them for the EKF
time = data['Time'].values  # Time in seconds (assuming it's in seconds; convert from ms if needed)
gyroData = data[['GyroX', 'GyroY', 'GyroZ']].values * np.pi / 180  # Convert degrees/sec to rad/sec if necessary
accData = data[['AccX', 'AccY', 'AccZ']].values * 9.81  # Accelerometer data
magData = data[['MagX', 'MagY', 'MagZ']].values * 100 # Magnetometer data

# Step 1: Calculate gyroscope bias
# Assuming the device was stationary at the start (adjust the index for your dataset)
# For example, you might want to consider the first 100 samples as stationary data
num_stationary_samples = 200
gyro_bias = np.mean(gyroData[:num_stationary_samples], axis=0)

# Step 2: Subtract bias from gyro data
gyroData_calibrated = gyroData - gyro_bias

# Assuming EKF is the Extended Kalman Filter implementation
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


def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w, x, y, z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


output = []
for i, q in enumerate(quaternions):
    roll, pitch, yaw = quaternion_to_euler(q)
    output.append(f"{i+1},{pitch:.2f},{roll:.2f},{yaw:.2f}")

# Print the output in CSV format
print("Id,pitch,roll,yaw")
for line in output:
    print(line)

data = {
    "Id": [i + 1 for i in range(len(quaternions))],
    "pitch": [quaternion_to_euler(q)[1] for q in quaternions],
    "roll": [quaternion_to_euler(q)[0] for q in quaternions],
    "yaw": [quaternion_to_euler(q)[2] for q in quaternions],
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save to CSV with two decimal places without modifying the DataFrame values
df.to_csv("orientation_test_output.csv", index=False, float_format="%.2f")


def plot_angles(computed_angles, csv_file):
    # Wczytaj dane z pliku CSV
    data = pd.read_csv(csv_file)
    
    # Wyodrębnij kolumny roll, pitch i yaw z danych
    roll_data = data['roll']
    pitch_data = data['pitch']
    yaw_data = data['yaw']

    computed = pd.read_csv("orientation_test_output.csv")
    roll_computed = computed['roll']
    pitch_computed = computed['pitch']
    yaw_computed = computed['yaw']
    
    # Wyodrębnij obliczone kąty
    # roll_computed = [angle[0] for angle in computed_angles]  # roll
    # pitch_computed = [angle[1] for angle in computed_angles]  # pitch
    # yaw_computed = [angle[2] for angle in computed_angles]    # yaw

    # Stwórz wykres
    plt.figure(figsize=(15, 8))
    
    # Wykres Roll
    plt.subplot(3, 1, 1)
    plt.plot(roll_data, label='Roll (CSV)', color='blue', linestyle='--', marker='o')
    plt.plot(roll_computed, label='Roll (Computed)', color='red')
    plt.title('Roll Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    # Wykres Pitch
    plt.subplot(3, 1, 2)
    plt.plot(pitch_data, label='Pitch (CSV)', color='blue', linestyle='--', marker='o')
    plt.plot(pitch_computed, label='Pitch (Computed)', color='red')
    plt.title('Pitch Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    # Wykres Yaw
    plt.subplot(3, 1, 3)
    plt.plot(yaw_data, label='Yaw (CSV)', color='blue', linestyle='--', marker='o')
    plt.plot(yaw_computed, label='Yaw (Computed)', color='red')
    plt.title('Yaw Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    # Wyświetl wykres
    plt.tight_layout()
    plt.show()


plot_angles(quaternions, 'data/train.csv')