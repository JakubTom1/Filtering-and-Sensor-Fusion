import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from EKF import IMUExtendedKalmanFilter


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
    N_CALIB = 9
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
    print(f"[INFO] Calculated Gyro Bias: {gyro_bias}")

    acc_mean_start = data[['AccX', 'AccY', 'AccZ']].iloc[:N_CALIB].mean().values

    acc_bias_correction = np.array([
        acc_mean_start[0] - 0.0,  # Expected X = 0
        acc_mean_start[1] - 0.0,  # Expected Y = 0 (Fixes Roll offset)
        acc_mean_start[2] - 1.0  # Expected Z = 1
    ])
    print(f"[INFO] Accelerometer Bias Correction: {acc_bias_correction}")

    # --- 2. Prepare Data Arrays ---

    # Apply Gyro Bias and Scale
    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE

    # Apply Accelerometer Bias
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values - acc_bias_correction

    # Magnetometer Data (Passed to EKF but ignored via high noise parameter)
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values

    times = data['Time'].values

    # Initialize EKF
    ekf = IMUExtendedKalmanFilter(
        process_noise=7.75e-5,  # Trust Gyro heavily
        accel_noise=0.005,  # Trust Accel moderately
        mag_noise=3  # Trust Mag moderately
    )

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


if __name__ == "__main__":
    run_benchmark()