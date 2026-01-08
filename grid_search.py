import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from EKF import IMUExtendedKalmanFilter

TRAIN_FILE = 'Data_contest_2/train.csv'
DT_DEFAULT = 0.005
N_CALIB = 200
GYRO_SCALE = 0.001 * (np.pi / 180.0)

# -------------------------------
# Load & preprocess data ONCE
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
# Grid definition
# -------------------------------
process_noise_grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
accel_noise_grid   = [0.01, 0.03, 0.05, 0.1, 0.2]
mag_noise_grid     = [0.1, 0.3, 0.5, 1.0, 2.0]

results = []

# -------------------------------
# Grid search
# -------------------------------
total_runs = len(process_noise_grid) * len(accel_noise_grid) * len(mag_noise_grid)
run_id = 1

for q in process_noise_grid:
    for r_acc in accel_noise_grid:
        for r_mag in mag_noise_grid:

            print(f"[{run_id}/{total_runs}] Q={q:.1e}, R_acc={r_acc}, R_mag={r_mag}")
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

            print(total_rmse)

            results.append([
                q, r_acc, r_mag,
                rmse_p, rmse_r, rmse_y,
                total_rmse
            ])

# -------------------------------
# Show best results
# -------------------------------
results = np.array(results, dtype=float)
results = results[results[:, -1].argsort()]

print("\n===== TOP 10 RESULTS =====")
for i in range(10):
    q, r_acc, r_mag, p, r, y, tot = results[i]
    print(
        f"{i+1:2d}: Q={q:.1e}, "
        f"R_acc={r_acc:.3f}, R_mag={r_mag:.3f} | "
        f"Pitch={p:.3f}, Roll={r:.3f}, Yaw={y:.3f} | Total={tot:.3f}"
    )
