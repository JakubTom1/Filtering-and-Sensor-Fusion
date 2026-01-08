import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from EKF import IMUExtendedKalmanFilter

# -------------------------------
# Constants
# -------------------------------
TRAIN_FILE = 'Data_contest_2/train.csv'
DT_DEFAULT = 0.005
N_CALIB = 200
GYRO_SCALE = 0.001 * (np.pi / 180.0)

# -------------------------------
# Load & preprocess data
# -------------------------------
print("Loading data...")
data = pd.read_csv(TRAIN_FILE)

gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
acc_mean = data[['AccX', 'AccY', 'AccZ']].iloc[:N_CALIB].mean().values

acc_bias = np.array([
    acc_mean[0],
    acc_mean[1],
    acc_mean[2] - 1.0
])

gyro = (data[['GyroX', 'GyroY', 'GyroZ']].values - gyro_bias) * GYRO_SCALE
acc = data[['AccX', 'AccY', 'AccZ']].values - acc_bias
mag = data[['MagX', 'MagY', 'MagZ']].values
time = data['Time'].values

truth_p = data['pitch'].values
truth_r = data['roll'].values
truth_y = data['yaw'].values

# -------------------------------
# Refined grid (≈100 runs)
# -------------------------------
process_noise_grid = np.logspace(np.log10(3e-5), np.log10(2e-4), 5)
accel_noise_grid   = [0.005, 0.01, 0.02, 0.03]
mag_noise_grid     = [0.8, 1.2, 1.6, 2.2, 3.0]

results = []
total_runs = len(process_noise_grid) * len(accel_noise_grid) * len(mag_noise_grid)
run_id = 1

# -------------------------------
# Grid search
# -------------------------------
for q in process_noise_grid:
    for r_acc in accel_noise_grid:
        for r_mag in mag_noise_grid:

            print(f"[{run_id}/{total_runs}] Q={q:.2e}, R_acc={r_acc}, R_mag={r_mag}")
            run_id += 1

            ekf = IMUExtendedKalmanFilter(
                process_noise=q,
                accel_noise=r_acc,
                mag_noise=r_mag
            )

            ekf.initialize_from_data(acc[0], mag[0])
            preds = []

            for i in range(len(time)):
                dt = DT_DEFAULT if i == 0 else time[i] - time[i - 1]
                if dt <= 0 or dt > 1.0:
                    dt = DT_DEFAULT

                ekf.predict(gyro[i], dt)
                ekf.update(acc[i], mag[i])

                roll, pitch, yaw = ekf.get_euler_angles()
                preds.append([pitch, roll, yaw])

            preds = np.array(preds)

            rmse_p = np.sqrt(mean_squared_error(truth_p, preds[:, 0]))
            rmse_r = np.sqrt(mean_squared_error(truth_r, preds[:, 1]))
            rmse_y = np.sqrt(mean_squared_error(truth_y, preds[:, 2]))

            total_rmse = rmse_p + rmse_r + rmse_y
            results.append([q, r_acc, r_mag, rmse_p, rmse_r, rmse_y, total_rmse])

# -------------------------------
# Results
# -------------------------------
results = np.array(results)
results = results[results[:, -1].argsort()]

print("\n===== TOP 10 REFINED RESULTS =====")
for i in range(10):
    q, r_acc, r_mag, p, r, y, tot = results[i]
    print(
        f"{i+1:2d}: Q={q:.2e}, R_acc={r_acc:.3f}, R_mag={r_mag:.3f} | "
        f"P={p:.3f}, R={r:.3f}, Y={y:.3f} | Total={tot:.3f}"
    )
