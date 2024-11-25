import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_angles(csv_file, reference_csv):
    # Load computed and reference data
    computed_data = pd.read_csv(csv_file)
    reference_data = pd.read_csv(reference_csv)

    roll_computed = computed_data['roll']
    pitch_computed = computed_data['pitch']
    yaw_computed = computed_data['yaw']

    roll_reference = reference_data['roll']
    pitch_reference = reference_data['pitch']
    yaw_reference = reference_data['yaw']

    plt.figure(figsize=(15, 8))

    # Roll comparison
    plt.subplot(3, 1, 1)
    plt.plot(roll_reference, label='Roll (Reference)', color='blue', linestyle='--')
    plt.plot(roll_computed, label='Roll (Computed)', color='red')
    plt.title('Roll Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    # Pitch comparison
    plt.subplot(3, 1, 2)
    plt.plot(pitch_reference, label='Pitch (Reference)', color='blue', linestyle='--')
    plt.plot(pitch_computed, label='Pitch (Computed)', color='red')
    plt.title('Pitch Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    # Yaw comparison
    plt.subplot(3, 1, 3)
    plt.plot(yaw_reference, label='Yaw (Reference)', color='blue', linestyle='--')
    plt.plot(yaw_computed, label='Yaw (Computed)', color='red')
    plt.title('Yaw Angles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('angles_comparison_filterpy.png')
    plt.show()


def calculate_mse(original_csv, computed_csv):
    original_data = pd.read_csv(original_csv)
    computed_data = pd.read_csv(computed_csv)

    mse_roll = mean_squared_error(original_data['roll'], computed_data['roll'])
    mse_pitch = mean_squared_error(original_data['pitch'], computed_data['pitch'])
    mse_yaw = mean_squared_error(original_data['yaw'], computed_data['yaw'])

    print(f"MSE for Roll: {mse_roll}")
    print(f"MSE for Pitch: {mse_pitch}")
    print(f"MSE for Yaw: {mse_yaw}")


