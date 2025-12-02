import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EKF import IMUExtendedKalmanFilter


def visualize_quality():
    print("--- WERYFIKACJA JAKOŚCI NA ZBIORZE TRENINGOWYM ---")
    TRAIN_FILE = 'Data/train.csv'

    try:
        data = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print("Brak pliku train.csv")
        return

    # Parametry (identyczne jak w submission)
    GYRO_SCALE = 0.001 * (np.pi / 180.0)
    N_CALIB = 200
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
    print(f"[INFO] Calculated Gyro Bias: {gyro_bias}")

    acc_mean_start = data[['AccX', 'AccY', 'AccZ']].iloc[:N_CALIB].mean().values

    acc_bias_correction = np.array([
        acc_mean_start[0] - 0.0,  # Expected X = 0
        acc_mean_start[1] - 0.0,  # Expected Y = 0 (Fixes Roll offset)
        acc_mean_start[2] - 1.0  # Expected Z = 1
    ])
    print(f"[INFO] Accelerometer Bias Correction: {acc_bias_correction}")

    # Apply Gyro Bias and Scale
    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE

    # Apply Accelerometer Bias
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values - acc_bias_correction

    # Magnetometer Data (Passed to EKF but ignored via high noise parameter)
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values
    times = data['Time'].values

    # EKF (Mag wyłączony)
    ekf = IMUExtendedKalmanFilter(process_noise=1e-5, accel_noise=0.05, mag_noise=1e6)
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    preds = []
    for i in range(len(times)):
        dt = 0.005 if i == 0 else times[i] - times[i - 1]
        if dt > 1.0 or dt <= 0: dt = 0.005
        ekf.predict(gyro_rad[i], dt)
        ekf.update(acc_vals[i], mag_vals[i])
        r, p, y = ekf.get_euler_angles()
        preds.append([p, r, y])

    preds = np.array(preds)

    # Rysowanie wykresów porównawczych
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    titles = ['Pitch (Pochylenie)', 'Roll (Przechył)', 'Yaw (Odchylenie)']
    truth_cols = ['pitch', 'roll', 'yaw']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i in range(3):
        axs[i].plot(times, data[truth_cols[i]], label='Rzeczywiste (Ground Truth)', color='black', linewidth=2,
                    alpha=0.5)
        axs[i].plot(times, preds[:, i], label='Estymacja EKF', color=colors[i], linestyle='--')
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Stopnie')
        axs[i].grid(True)
        axs[i].legend()

    plt.xlabel('Czas [s]')
    plt.tight_layout()
    plt.savefig('quality_check.png')
    plt.show()
    print("Wykres porównawczy zapisano jako 'quality_check.png'. Sprawdź pokrycie linii.")


if __name__ == "__main__":
    visualize_quality()