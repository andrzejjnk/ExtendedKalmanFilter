import numpy as np
import pandas as pd
import csv
import math
from EKF import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Wczytywanie danych
data = pd.read_csv('data/train.csv')

# Ekstrakcja danych dla czujników
time = data['Time'].values  # Zakładamy, że czas jest w sekundach
gyroData = data[['GyroX', 'GyroY', 'GyroZ']].values * np.pi / 180  # Zmiana jednostek z deg/s na rad/s
accData = data[['AccX', 'AccY', 'AccZ']].values * 9.81  # Zmiana jednostek na m/s²
magData = data[['MagX', 'MagY', 'MagZ']].values * 1000  # Zmiana jednostek magnetometru (jeśli dane są w mG)

# Określenie liczby próbek w stanie spoczynku
num_stationary_samples = 400

# Krok 1: Kalibracja żyroskopu
gyro_bias = np.mean(gyroData[:1000], axis=0)
gyroData_calibrated = gyroData - gyro_bias

# Krok 2: Kalibracja akcelerometru
# Zakładamy, że w stanie spoczynku przyspieszenie w osi Z = 9.81, a w osiach X i Y = 0
acc_bias = np.mean(accData[:1000], axis=0) - np.array([0, 0, 9.81])
accData_calibrated = accData - acc_bias

# Krok 3: Kalibracja magnetometru
# Zakładamy przesunięcie jako średnią wartość początkowych próbek w każdej osi
mag_bias = np.mean(magData[:num_stationary_samples], axis=0)
magData_calibrated = magData - mag_bias

# Inicjalizacja EKF z poprawionymi danymi (należy mieć zaimplementowany EKF)
ekf = EKF(gyroData=gyroData_calibrated, accData=accData_calibrated, time=time)
ekf.P *= 1  # Opcjonalna modyfikacja macierzy kowariancji na początek

# Uruchomienie EKF
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

data = {
    "Id": [i + 1 for i in range(len(quaternions))],
    "pitch": [quaternion_to_euler(q)[1] for q in quaternions],
    "roll": [quaternion_to_euler(q)[0] for q in quaternions],
    "yaw": [quaternion_to_euler(q)[2] for q in quaternions],
}

df = pd.DataFrame(data)
df.to_csv("orientation_test_output.csv", index=False, float_format="%.2f")


def plot_angles(csv_file):
    data = pd.read_csv(csv_file)
    
    roll_data = data['roll']
    pitch_data = data['pitch']
    yaw_data = data['yaw']

    computed = pd.read_csv("orientation_test_output.csv")
    roll_computed = computed['roll']
    pitch_computed = computed['pitch']
    yaw_computed = computed['yaw']
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(roll_data, label='Roll (Train)', color='blue', linestyle='--', marker='o')
    plt.plot(roll_computed, label='Roll (Computed)', color='red')
    plt.title('Roll Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(pitch_data, label='Pitch (Train)', color='blue', linestyle='--', marker='o')
    plt.plot(pitch_computed, label='Pitch (Computed)', color='red')
    plt.title('Pitch Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(yaw_data, label='Yaw (Train)', color='blue', linestyle='--', marker='o')
    plt.plot(yaw_computed, label='Yaw (Computed)', color='red')
    plt.title('Yaw Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


plot_angles('data/train.csv')

def calculate_mse(original_csv, computed_csv):
    # Load original and computed data
    original_data = pd.read_csv(original_csv)
    computed_data = pd.read_csv(computed_csv)
    
    # Extract roll, pitch, yaw from each dataset
    roll_original = original_data['roll']
    pitch_original = original_data['pitch']
    yaw_original = original_data['yaw']
    
    roll_computed = computed_data['roll']
    pitch_computed = computed_data['pitch']
    yaw_computed = computed_data['yaw']
    
    # Calculate mean squared error for each angle
    mse_roll = mean_squared_error(roll_original, roll_computed)
    mse_pitch = mean_squared_error(pitch_original, pitch_computed)
    mse_yaw = mean_squared_error(yaw_original, yaw_computed)
    
    # Print and return the results
    print(f"MSE for Roll: {mse_roll}")
    print(f"MSE for Pitch: {mse_pitch}")
    print(f"MSE for Yaw: {mse_yaw}")
    
    return mse_roll, mse_pitch, mse_yaw

# Example usage:
mse_roll, mse_pitch, mse_yaw = calculate_mse("data/train.csv", "orientation_test_output.csv")