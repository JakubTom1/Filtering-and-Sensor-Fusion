import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EKF import IMUExtendedKalmanFilter


def generate_submission_and_plot():
    """
    Main script to process the test dataset using the tuned EKF.

    Workflow:
    1. Loads 'test.csv'.
    2. Performs static calibration for Gyroscope (drift removal) and Accelerometer (leveling).
    3. Runs the EKF loop.
    4. Saves the results to 'submission.csv'.
    5. Generates a plot 'submission_plot.png' for visual verification.
    """

    # File paths
    TEST_FILE = 'Data/test.csv'
    OUT_FILE = 'Data/submission.csv'
    PLOT_FILE = 'Data/submission_plot.png'

    print(f"[INFO] Loading data from {TEST_FILE}...")
    try:
        data = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File {TEST_FILE} not found. Please upload it.")
        return

    # --- 1. Preprocessing & Calibration ---

    # Unit conversion factor: milli-degrees/s -> radians/s
    GYRO_SCALE = 0.001 * (np.pi / 180.0)

    # Calibration Samples: Use the first 200 samples (assuming the drone is stationary at start)
    N_CALIB = 200

    # A. Gyroscope Calibration (Zero-rate offset)
    # We calculate the mean value when stationary and subtract it from all readings.
    gyro_bias = data[['GyroX', 'GyroY', 'GyroZ']].iloc[:N_CALIB].mean().values
    print(f"[INFO] Calculated Gyro Bias: {gyro_bias}")

    # B. Accelerometer Calibration (Leveling Correction)
    # Goal: Fix the "Roll" offset.
    # Assumption: At start, the drone is flat. Ideally: AccX=0, AccY=0, AccZ=1g.
    # We calculate the deviation from this ideal state and apply it as a bias.
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
    ids = data['Id'].values

    # --- 3. Initialize EKF ---

    # Tuning parameters based on training data analysis:
    # - process_noise: 1e-5 (High trust in Gyroscope integration)
    # - accel_noise: 0.05   (Moderate trust in Gravity vector)
    # - mag_noise: 1e6      (Disable Magnetometer updates due to unreliable data)
    ekf = IMUExtendedKalmanFilter(
        process_noise=1e-5,
        accel_noise=0.05,
        mag_noise=1e6
    )

    # Initialize State (Forces Yaw=0.0 and calculates initial Pitch/Roll from Accel)
    ekf.initialize_from_data(acc_vals[0], mag_vals[0])

    results = []

    # Lists for plotting
    plot_pitch, plot_roll, plot_yaw = [], [], []

    print("[INFO] Running EKF Loop...")

    # --- 4. Main Processing Loop ---
    for i in range(len(times)):
        # Calculate time step (dt)
        if i == 0:
            dt = 0.005  # Default sampling time (approx 200Hz)
        else:
            dt = times[i] - times[i - 1]
            # Handle potential data gaps or timestamp errors
            if dt > 1.0 or dt <= 0: dt = 0.005

        # Predict Step (Gyro Integration)
        ekf.predict(gyro_rad[i], dt)

        # Update Step (Accel Correction)
        ekf.update(acc_vals[i], mag_vals[i])

        # Extract Euler Angles
        r, p, y = ekf.get_euler_angles()

        # Store results (Format: Id, Pitch, Roll, Yaw)
        results.append([int(ids[i]), p, r, y])

        # Store for plotting
        plot_pitch.append(p)
        plot_roll.append(r)
        plot_yaw.append(y)

    # --- 5. Save Results ---
    df_out = pd.DataFrame(results, columns=['Id', 'pitch', 'roll', 'yaw'])
    df_out.to_csv(OUT_FILE, index=False)
    print(f"[SUCCESS] Submission saved to '{OUT_FILE}'")

    # --- 6. Visualization ---
    print(f"[INFO] Generating trajectory plot to '{PLOT_FILE}'...")

    plt.figure(figsize=(12, 10))

    # Pitch Subplot
    plt.subplot(3, 1, 1)
    plt.plot(times, plot_pitch, color='blue', linewidth=1)
    plt.title('Predicted Pitch (Degrees)')
    plt.ylabel('Angle [deg]')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Roll Subplot
    plt.subplot(3, 1, 2)
    plt.plot(times, plot_roll, color='orange', linewidth=1)
    plt.title('Predicted Roll (Degrees) - Corrected Leveling')
    plt.ylabel('Angle [deg]')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Yaw Subplot
    plt.subplot(3, 1, 3)
    plt.plot(times, plot_yaw, color='green', linewidth=1)
    plt.title('Predicted Yaw (Degrees) - Gyro Integration Only')
    plt.ylabel('Angle [deg]')
    plt.xlabel('Time [s]')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.show()
    plt.close()  # Close plot to free memory
    print("[DONE] Process completed.")


if __name__ == "__main__":
    generate_submission_and_plot()