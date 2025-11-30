import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from EKF import IMUExtendedKalmanFilter


def get_mag_hard_iron_bias(calib_file):
    try:
        df = pd.read_csv(calib_file)
        mag_cols = ['MagX', 'MagY', 'MagZ']
        return ((df[mag_cols].min() + df[mag_cols].max()) / 2.0).values
    except:
        return np.zeros(3)


def run_benchmark():
    TRAIN_FILE = 'Data/train.csv'
    CALIB_FILE = 'Data/calib.csv'

    data = pd.read_csv(TRAIN_FILE)

    # Config
    GYRO_SCALE = 0.001

    # Bias
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:200].mean().values
    mag_bias = get_mag_hard_iron_bias(CALIB_FILE)

    # Data Prep
    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE * (np.pi / 180.0)
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values - mag_bias
    times = data['Time'].values

    # EKF
    ekf = IMUExtendedKalmanFilter()
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    preds = []

    for i in range(len(times)):
        dt = 0.005 if i == 0 else times[i] - times[i - 1]

        ekf.predict(gyro_rad[i], dt)
        ekf.update(acc_vals[i], mag_vals[i])

        r, p, y = ekf.get_euler_angles()
        preds.append([p, r, y])

    preds = np.array(preds)

    # RMSE
    rmse_pitch = np.sqrt(mean_squared_error(data['pitch'], preds[:, 0]))
    rmse_roll = np.sqrt(mean_squared_error(data['roll'], preds[:, 1]))
    rmse_yaw = np.sqrt(mean_squared_error(data['yaw'], preds[:, 2]))

    print("-" * 30)
    print(f"Benchmark Results (RMSE):")
    print(f"Pitch RMSE: {rmse_pitch:.4f} degrees")
    print(f"Roll  RMSE: {rmse_roll:.4f} degrees")
    print(f"Yaw   RMSE: {rmse_yaw:.4f} degrees")
    print("-" * 30)

    total_error = rmse_pitch + rmse_roll + rmse_yaw
    print(f"Total Cumulative Error: {total_error:.4f}")
    print("If Total Error is low (< 10.0), the filter is tuned reasonably well.")

if __name__ == "__main__":
    run_benchmark()