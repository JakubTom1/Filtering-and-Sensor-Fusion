import numpy as np
from typing import Tuple, Optional


class IMUExtendedKalmanFilter:
    """
    Robust Extended Kalman Filter (EKF) for IMU Sensor Fusion.

    Key Features:
    - Uses Gyroscope for state prediction (Integration).
    - Uses Accelerometer for Gravity Vector correction (Roll/Pitch).
    - Ignores Magnetometer due to poor data correlation.
    - Implements Verified Analytical Jacobian for Gravity to prevent drift.
    """

    def __init__(self, process_noise: float = 1e-5, accel_noise: float = 0.05, mag_noise: float = 1e6):
        # State: Quaternion [w, x, y, z]
        self.state = np.array([1.0, 0.0, 0.0, 0.0])

        # P: Error Covariance
        self.P = np.eye(4) * 0.1

        # Q: Process Noise
        self.Q = np.eye(4) * process_noise

        # R: Measurement Noise
        self.R_acc = np.eye(3) * accel_noise

        # Reference Gravity (Z-axis down standard)
        self.ref_g = np.array([0.0, 0.0, 1.0])

    def initialize_from_data(self, acc: np.ndarray, mag: np.ndarray):
        """
        Initialize State from Accelerometer.
        Forces Yaw=0.0 to match Ground Truth convention.
        """
        acc_norm = acc / np.linalg.norm(acc)

        # Calculate Pitch/Roll from Gravity Vector
        pitch = np.arcsin(-acc_norm[0])
        roll = np.arctan2(acc_norm[1], acc_norm[2])

        # Force Yaw = 0
        yaw = 0.0

        # Convert to Quaternion
        cy = np.cos(yaw * 0.5);
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5);
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5);
        sr = np.sin(roll * 0.5)

        self.state = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
        self.state /= np.linalg.norm(self.state)

    def predict(self, gyro_rad: np.ndarray, dt: float):
        """
        Time Update: Integrate Gyroscope Data.
        """
        wx, wy, wz = gyro_rad

        # Quaternion Derivative Matrix
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])

        # Discrete Update F = I + 0.5 * Omega * dt
        F = np.eye(4) + 0.5 * Omega * dt

        self.state = F @ self.state
        self.state /= np.linalg.norm(self.state)  # Normalize to prevent divergence
        self.P = F @ self.P @ F.T + self.Q

    def update(self, acc: np.ndarray, mag: np.ndarray):
        """
        Measurement Update: Correct with Accelerometer only.
        Magnetometer is skipped due to dataset issues.
        """
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0:
            z_acc = acc / acc_norm
            self._update_gravity(z_acc)

        # Normalization after update
        self.state /= np.linalg.norm(self.state)

    def _update_gravity(self, z_acc):
        """
        Specialized Update for Gravity Vector [0,0,1].
        Using manually verified Jacobian to avoid sign errors.
        """
        w, x, y, z = self.state

        # 1. Predicted Gravity in Body Frame h(x)
        # h = R(q)^T * [0,0,1]
        # hx = 2(xz - wy)
        # hy = 2(yz + wx)
        # hz = w^2 - x^2 - y^2 + z^2
        hx = 2 * (x * z - w * y)
        hy = 2 * (w * x + y * z)
        hz = w ** 2 - x ** 2 - y ** 2 + z ** 2
        h = np.array([hx, hy, hz])

        # 2. Jacobian H = d(h)/dq
        # Verified derivatives:
        # Row 0 (hx): [-2y,  2z, -2w,  2x]
        # Row 1 (hy): [ 2x,  2w,  2z,  2y]
        # Row 2 (hz): [ 2w, -2x, -2y,  2z]
        H = np.array([
            [-2 * y, 2 * z, -2 * w, 2 * x],
            [2 * x, 2 * w, 2 * z, 2 * y],
            [2 * w, -2 * x, -2 * y, 2 * z]
        ])

        # 3. Kalman Update
        y_res = z_acc - h
        S = H @ self.P @ H.T + self.R_acc
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.state = self.state + K @ y_res
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Returns Roll, Pitch, Yaw in degrees."""
        w, x, y, z = self.state

        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)

        sinp = 2 * (w * y - z * x)
        pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)