import numpy as np
from typing import Tuple

class IMUExtendedKalmanFilter:
    """
    Extended Kalman Filter for IMU with:
    - Quaternion state [q_w, q_x, q_y, q_z]
    - Gyroscope biases [b_gx, b_gy, b_gz]
    - Gravity correction via accelerometer
    - Yaw correction via magnetometer (with offset)
    """


    #param 4.61 - process_noise=1e-5, accel_noise=0.05, mag_noise=0.5

    def __init__(self, process_noise=1e-5, accel_noise=0.05, mag_noise=0.5):
        # Stan: [q_w, q_x, q_y, q_z, b_gx, b_gy, b_gz]
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(7) * 0.1  # Kowariancja stanu

        self.Q = np.eye(7) * process_noise
        self.R_acc = np.eye(3) * accel_noise
        self.R_mag = np.array([[mag_noise]])

        # Yaw offset (rad) obliczony z danych treningowych
        self.yaw_offset = np.deg2rad(104.28)

    # --------------------------------------------------
    def initialize_from_data(self, acc, mag):
        acc = acc / np.linalg.norm(acc)
        pitch = np.arcsin(-acc[0])
        roll = np.arctan2(acc[1], acc[2])
        yaw = 0.0

        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cr, sr = np.cos(roll/2), np.sin(roll/2)

        # Zaktualizuj tylko quaternion w stanie
        self.state[:4] = np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])
        self.state[:4] /= np.linalg.norm(self.state[:4])
        # Biasy gyro pozostają 0

    # --------------------------------------------------
    def predict(self, gyro_rad, dt):
        # Odejmij biasy od odczytu gyro
        wx, wy, wz = gyro_rad - self.state[4:7]

        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])

        # Dyskretny update quaternion
        Fq = np.eye(4) + 0.5 * Omega * dt
        q = self.state[:4]
        q = Fq @ q
        q /= np.linalg.norm(q)
        self.state[:4] = q

        # Macierz przejścia dla całego stanu (quaternion + bias)
        F = np.eye(7)
        F[:4, :4] = Fq  # quaternion
        # Biasy: random walk (F[4:,4:] = I)

        # Kowariancja stanu
        self.P = F @ self.P @ F.T + self.Q * dt**2

    # --------------------------------------------------
    def update(self, acc, mag):
        if np.linalg.norm(acc) > 0:
            self._update_gravity(acc / np.linalg.norm(acc))

        if np.linalg.norm(mag) > 0:
            self._update_yaw_mag(mag)

        # Normalizacja quaternion
        self.state[:4] /= np.linalg.norm(self.state[:4])

    # --------------------------------------------------
    def _update_gravity(self, z_acc):
        w, x, y, z = self.state[:4]

        # Predykcja grawitacji w ramce ciała
        h = np.array([
            2*(x*z - w*y),
            2*(w*x + y*z),
            w*w - x*x - y*y + z*z
        ])

        # Jacobian H (3x7)
        H = np.zeros((3,7))
        H[:, :4] = np.array([
            [-2*y,  2*z, -2*w, 2*x],
            [ 2*x,  2*w,  2*z, 2*y],
            [ 2*w, -2*x, -2*y, 2*z]
        ])

        y_res = z_acc - h
        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state += K @ y_res
        self.P = (np.eye(7) - K @ H) @ self.P

    # --------------------------------------------------
    def _update_yaw_mag(self, mag):
        mag = mag / np.linalg.norm(mag)
        mx, my, mz = mag

        roll, pitch, yaw_pred = self.get_euler_angles()
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw_pred = np.radians(yaw_pred)

        # Tilt compensation
        mx2 = mx*np.cos(pitch) + mz*np.sin(pitch)
        my2 = mx*np.sin(roll)*np.sin(pitch) + my*np.cos(roll) - mz*np.sin(roll)*np.cos(pitch)
        yaw_meas = np.arctan2(-my2, mx2) + self.yaw_offset
        yaw_meas = (yaw_meas + np.pi) % (2*np.pi) - np.pi

        # Innovation
        y_res = yaw_meas - yaw_pred
        y_res = (y_res + np.pi) % (2*np.pi) - np.pi

        # Innovation gate
        if abs(y_res) > np.deg2rad(30):
            return

        w, x, y, z = self.state[:4]
        H = np.zeros((1,7))
        H[0,:4] = [-2*z, -2*y, 2*x, 2*w]

        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state += (K.flatten() * y_res)
        self.P = (np.eye(7) - K @ H) @ self.P

    # --------------------------------------------------
    def get_euler_angles(self) -> Tuple[float, float, float]:
        w, x, y, z = self.state[:4]

        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
