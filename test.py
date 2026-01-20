import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from EKF import IMUExtendedKalmanFilter
import matplotlib.pyplot as plt

# KONFIGURACJA: ustaw na True aby wygenerować submission zamiast uruchamiać benchmark
GENERATE_SUBMISSION = True

# Domyślne parametry EKF używane zarówno w benchmarku jak i przy generowaniu submission
EKF_PARAMS = dict(
    process_noise=4e-5,
    accel_noise=0.007,
    mag_noise=35
)

N_CALIB = 9 # Liczba próbek do kalibracji na początku
ACC_BIAS_CORRECTION = [-0.01, 0.004, -0.025]  # Korekta biasu akcelerometru w benchmarku
#ACC_BIAS_CORRECTION = [-0.007, 0, -0.02]  # Korekta biasu akcelerometru w benchmarku
GYRO_BIAS = [25, -70, -0.5]  # Dodatkowy bias żyroskopu (jeśli potrzebny)

def run_benchmark():
    TRAIN_FILE = 'Data_contest_2/train.csv'
    print("Loading data...")

    try:
        data = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print("Error: train.csv not found.")
        return

    # Scaling: milli-degrees/s -> rad/s
    GYRO_SCALE = 0.001 * (np.pi / 180.0)
    
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
    gyro_bias = np.array([
        gyro_bias[0]  + GYRO_BIAS[0],
        gyro_bias[1] + GYRO_BIAS[1],
        gyro_bias[2] + GYRO_BIAS[2]  
    ])
    #print("head of train file: ")
    #print(data[:20])
    print(f"[INFO] Calculated Gyro Bias: {gyro_bias}")

    acc_mean_start = data[['AccX', 'AccY', 'AccZ']].iloc[:N_CALIB].mean().values

    acc_bias_correction = np.array([
        acc_mean_start[0] - 0.0 + ACC_BIAS_CORRECTION[0],  # Expected X = 0
        acc_mean_start[1] - 0.0 + ACC_BIAS_CORRECTION[1],  # Expected Y = 0 (Fixes Roll offset)
        acc_mean_start[2] - 1.0 + ACC_BIAS_CORRECTION[2]  # Expected Z = 1
    ])
    print(f"[INFO] Accelerometer Bias Correction: {acc_bias_correction}")

    # --- 2. Prepare Data Arrays ---
    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values - acc_bias_correction
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values
    times = data['Time'].values

    # Initialize EKF
    ekf = IMUExtendedKalmanFilter(**EKF_PARAMS)
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    preds = []

    print("Running EKF...")
    for i in range(len(times)):
        dt = 0.005 if i == 0 else times[i] - times[i - 1]
        if dt > 1.0 or dt <= 0: dt = 0.005

        ekf.predict(gyro_rad[i], dt)
        ekf.update(acc_vals[i], mag_vals[i])

        r, p, y = ekf.get_euler_angles()
        preds.append([p, r, y])

    preds = np.array(preds)

    # Calculate RMSE
    truth_p = data['pitch'].values
    truth_r = data['roll'].values
    truth_y = data['yaw'].values

    rmse_p = np.sqrt(mean_squared_error(truth_p, preds[:, 0]))
    rmse_r = np.sqrt(mean_squared_error(truth_r, preds[:, 1]))
    rmse_y = np.sqrt(mean_squared_error(truth_y, preds[:, 2]))

    print("-" * 30)
    print(f"Benchmark Results (RMSE):")
    print(f"Pitch: {rmse_p:.4f} deg")
    print(f"Roll : {rmse_r:.4f} deg")
    print(f"Yaw  : {rmse_y:.4f} deg")
    print("-" * 30)
    print(f"Total: {rmse_p + rmse_r + rmse_y:.4f}")

def generate_submission():
    TEST_FILE = 'Data_contest_2/test.csv'
    OUT_FILE = 'Data_contest_2/submission.csv'
    TRAIN_FILE= 'Data_contest_2/train.csv'
    print(f"[INFO] Loading data from {TEST_FILE}...")

    try:
        data = pd.read_csv(TEST_FILE)
        data_train = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File {TEST_FILE} not found.")
        return

    GYRO_SCALE = 0.001 * (np.pi / 180.0)

    # Calibration using first N_CALIB samples (assume stationary)
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
    gyro_bias = np.array([
        gyro_bias[0]  + GYRO_BIAS[0],
        gyro_bias[1] + GYRO_BIAS[1],
        gyro_bias[2] + GYRO_BIAS[2]  
    ])
    print(f"[INFO] Calculated Gyro Bias: {gyro_bias}")

    acc_mean_start = data[['AccX', 'AccY', 'AccZ']].iloc[:N_CALIB].mean().values
    acc_bias_correction = np.array([
        acc_mean_start[0] - 0.0 + ACC_BIAS_CORRECTION[0],  # Expected X = 0
        acc_mean_start[1] - 0.0 + ACC_BIAS_CORRECTION[1],  # Expected Y = 0 (Fixes Roll offset)
        acc_mean_start[2] - 1.0 + ACC_BIAS_CORRECTION[2]  # Expected Z = 1
    ])
    print(f"[INFO] Accelerometer Bias Correction: {acc_bias_correction}")

    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values - acc_bias_correction
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values
    times = data['Time'].values
    ids = data['Id'].values

    ekf = IMUExtendedKalmanFilter(**EKF_PARAMS)
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    results = []
    print("[INFO] Running EKF Loop for submission...")

    for i in range(len(times)):
        dt = 0.005 if i == 0 else times[i] - times[i - 1]
        if dt > 1.0 or dt <= 0: dt = 0.005

        ekf.predict(gyro_rad[i], dt)
        ekf.update(acc_vals[i], mag_vals[i])

        r, p, y = ekf.get_euler_angles()
        results.append([int(ids[i]), p, r, y])

    df_out = pd.DataFrame(results, columns=['Id', 'pitch', 'roll', 'yaw'])
    df_out.to_csv(OUT_FILE, index=False)
    print(f"[SUCCESS] Submission saved to '{OUT_FILE}'")

   # ===== WYKRES: TYLKO TEST (EKF) =====
    results = np.array(results)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    titles = ['Pitch (Pochylenie)', 'Roll (Przechył)', 'Yaw (Odchylenie)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i in range(3):
        axs[i].plot(
            times,
            results[:, i + 1],
            color=colors[i],
            label='EKF'
        )
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Stopnie')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel('Czas [s]')
    plt.tight_layout()
    plt.savefig('quality_check_test.png')
    plt.show()

    print("[INFO] Wykres testowy zapisany jako quality_check_test.png")


if __name__ == "__main__":
    if GENERATE_SUBMISSION:
        generate_submission()
    else:
        run_benchmark()