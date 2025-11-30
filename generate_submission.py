import pandas as pd
import numpy as np
from EKF import IMUExtendedKalmanFilter


def get_mag_hard_iron_bias(calib_file):
    """Calculates Hard Iron offset from calibration file using Min/Max center."""
    try:
        df = pd.read_csv(calib_file)
        mag_cols = ['MagX', 'MagY', 'MagZ']
        # Center = (Min + Max) / 2
        bias = (df[mag_cols].min() + df[mag_cols].max()) / 2.0
        return bias.values
    except FileNotFoundError:
        return np.zeros(3)


def process():
    TEST_FILE = 'Data/test.csv'
    CALIB_FILE = 'Data/calib.csv'
    OUT_FILE = 'Data/submission.csv'

    # 1. Load Data
    try:
        data = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print("Test file not found.")
        return

    # 2. Sensor Config
    # GYRO_SCALE: Raw units are millidegrees/s. Convert to deg/s.
    GYRO_SCALE = 0.001

    # Gyro Bias: Calculate from start of TEST file (assume stationary start)
    # Using first 200 samples
    gyro_bias_raw = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:200].mean().values

    # Mag Bias: Calculate from CALIB file (Hard Iron)
    mag_bias = get_mag_hard_iron_bias(CALIB_FILE)

    # Prepare Arrays
    # Convert Raw Gyro -> Deg/s -> Rad/s
    gyro_rad = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias_raw) * GYRO_SCALE * (np.pi / 180.0)
    acc_vals = data[['AccX', 'AccY', 'AccZ']].values
    mag_vals = data[['MagX', 'MagY', 'MagZ']].values - mag_bias  # Apply Hard Iron Correction

    times = data['Time'].values
    ids = data['Id'].values

    # 3. Run EKF
    ekf = IMUExtendedKalmanFilter(process_noise=1e-5, accel_noise=0.01, mag_noise=0.1)

    results = []

    # Initialize with first sample
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    for i in range(len(times)):
        if i == 0:
            dt = 0.005
        else:
            dt = times[i] - times[i - 1]
            if dt > 1.0 or dt <= 0: dt = 0.005

        ekf.predict(gyro_rad[i], dt)
        ekf.update(acc_vals[i], mag_vals[i])

        r, p, y = ekf.get_euler_angles()
        results.append([int(ids[i]), p, r, y])

    # 4. Save
    pd.DataFrame(results, columns=['Id', 'pitch', 'roll', 'yaw']).to_csv(OUT_FILE, index=False)
    print("Done. Saved submission.csv")


if __name__ == "__main__":
    process()