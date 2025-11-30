import numpy as np
import math
from typing import Tuple, Optional


class IMUExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for IMU Sensor Fusion — improved version.

    Zmiany kluczowe:
    - inicjalizacja yaw z kompensacją przechyłu (tilt compensation),
    - numeryczny Jacobian (finite differences) dla pomiarów (acc/mag),
    - Joseph-form update kowariancji,
    - skalowanie Q zależnie od dt oraz zabezpieczenia normowania.
    """

    def __init__(self, process_noise: float = 1e-5, accel_noise: float = 0.01, mag_noise: float = 0.1):
        # State Vector: Quaternion [q_w, q_x, q_y, q_z]
        self.state = np.array([1.0, 0.0, 0.0, 0.0])

        # Covariance Matrix P (4x4 for quaternion parameters; note: quaternion constraint handled by renormalization)
        self.P = np.eye(4) * 0.1

        # Base process noise (will be scaled by dt)
        self.base_process_noise = process_noise
        self.Q = np.eye(4) * process_noise

        # Measurement Noise R
        self.R_acc = np.eye(3) * accel_noise
        self.R_mag = np.eye(3) * mag_noise

        # Reference Vectors (World Frame)
        self.ref_g = np.array([0.0, 0.0, 1.0])  # Gravity reference (world)
        self.ref_m: Optional[np.ndarray] = None  # Magnetic reference (world), ustawione przy inicjalizacji

    # ---------------------------
    # Initialization helpers
    # ---------------------------
    def initialize_from_data(self, acc: np.ndarray, mag: np.ndarray):
        """
        Initialize quaternion state using accelerometer + magnetometer.
        Performs tilt-compensation of magnetometer to get initial yaw.
        """
        if np.linalg.norm(acc) == 0 or np.linalg.norm(mag) == 0:
            # fallback to identity quaternion
            self.state = np.array([1.0, 0.0, 0.0, 0.0])
            self.ref_m = np.array([1.0, 0.0, 0.0])
            return

        acc_norm = acc / np.linalg.norm(acc)
        mag_norm = mag / np.linalg.norm(mag)

        # Roll and Pitch from accelerometer (assuming gravity dominant)
        # roll: rotation about X, pitch: rotation about Y
        pitch = np.arcsin(np.clip(-acc_norm[0], -1.0, 1.0))
        roll = np.arctan2(acc_norm[1], acc_norm[2])

        # Tilt-compensate magnetometer to obtain yaw
        # Following standard tilt compensation formula:
        sin_r = np.sin(roll); cos_r = np.cos(roll)
        sin_p = np.sin(pitch); cos_p = np.cos(pitch)

        # Rotate mag into horizontal frame
        mx = mag_norm[0]; my = mag_norm[1]; mz = mag_norm[2]
        # Compensated components (body -> horizontal)
        mag_x_h = mx * cos_p + mz * sin_p
        mag_y_h = mx * sin_r * sin_p + my * cos_r - mz * sin_r * cos_p

        yaw = np.arctan2(-mag_y_h, mag_x_h)  # sign convention chosen to match rotation matrix below

        # Build quaternion from roll, pitch, yaw
        cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)

        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy

        self.state = np.array([q0, q1, q2, q3])
        self.state = self.state / np.linalg.norm(self.state)

        # Set magnetic reference in WORLD frame: ref_m = R_body_to_world * mag_body
        R = self._get_rotation_matrix(self.state)
        self.ref_m = R @ mag_norm

    # ---------------------------
    # Predict step
    # ---------------------------
    def predict(self, gyro_rad: np.ndarray, dt: float) -> None:
        """
        Prediction Step (Gyro Integration).
        gyro_rad: [wx, wy, wz] in radians/second.
        """
        if dt <= 0:
            return

        wx, wy, wz = gyro_rad

        # Omega matrix for quaternion derivative: q_dot = 0.5 * Omega(omega) @ q
        omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0,  wx],
            [wz,  wy, -wx,  0.0]
        ])

        # Discrete first-order integration: q_{k+1} ≈ (I + 0.5 * Omega * dt) q_k
        F = np.eye(4) + 0.5 * omega * dt

        # Predict state
        self.state = F @ self.state
        self.state = self.state / np.linalg.norm(self.state)

        # Scale process noise with dt (simple heuristic)
        self.Q = np.eye(4) * (self.base_process_noise * max(dt, 1e-6))

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        # Ensure symmetry / numerical stability
        self.P = 0.5 * (self.P + self.P.T)

    # ---------------------------
    # Update (correction)
    # ---------------------------
    def update(self, acc: np.ndarray, mag: np.ndarray) -> None:
        """
        Correction Step: fuse accelerometer and magnetometer.
        """
        acc_norm = np.linalg.norm(acc)
        mag_norm = np.linalg.norm(mag)
        if acc_norm == 0 or mag_norm == 0:
            return

        z_acc = acc / acc_norm
        z_mag = mag / mag_norm

        if self.ref_m is None:
            self.initialize_from_data(acc, mag)

        # Correct using Gravity measurement (world->body expected)
        self._correct(z_acc, self.ref_g, self.R_acc)

        # Correct using Magnetic field measurement
        self._correct(z_mag, self.ref_m, self.R_mag)

    # ---------------------------
    # Generic correction with numerical Jacobian
    # ---------------------------
    def _correct(self, z_meas: np.ndarray, v_ref: np.ndarray, R_cov: np.ndarray):
        """
        Generic Kalman update for a 3D vector observation.
        Measurement model: h(q) = R(q)^T @ v_ref  (world->body)
        We compute Jacobian numerically (finite differences) to avoid analytic sign mistakes.
        """
        # Ensure normalized quaternion
        q = self.state / np.linalg.norm(self.state)
        q0, q1, q2, q3 = q

        # Expected measurement
        R = self._get_rotation_matrix(q)   # body->world
        h_x = R.T @ v_ref                 # predicted measurement in body frame

        # Numerical Jacobian: small perturbations on quaternion components (with renormalization)
        eps = 1e-6
        H = np.zeros((3, 4))
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q_pert = q + dq
            q_pert = q_pert / np.linalg.norm(q_pert)
            R_pert = self._get_rotation_matrix(q_pert)
            h_pert = R_pert.T @ v_ref
            H[:, i] = (h_pert - h_x) / eps

        # Innovation / residual
        y = z_meas - h_x

        # Innovation covariance
        S = H @ self.P @ H.T + R_cov

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # fallback: skip update if S is singular
            return

        # State addition: Add correction to quaternion
        dq_state = K @ y
        q_new = q + dq_state
        # renormalize
        q_new = q_new / np.linalg.norm(q_new)
        self.state = q_new

        # Joseph form covariance update for numerical stability
        I = np.eye(4)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R_cov @ K.T

        # Force symmetry
        self.P = 0.5 * (self.P + self.P.T)

    # ---------------------------
    # Rotation matrix from quaternion (body -> world)
    # ---------------------------
    def _get_rotation_matrix(self, q):
        """Body to World Rotation Matrix from Quaternion q = [w,x,y,z]"""
        w, x, y, z = q
        # Note: this returns rotation that maps a vector in body frame to world frame:
        # v_world = R_body_to_world @ v_body
        return np.array([
            [1 - 2 * (y ** 2 + z ** 2),     2 * (x * y - w * z),         2 * (x * z + w * y)],
            [2 * (x * y + w * z),           1 - 2 * (x ** 2 + z ** 2),   2 * (y * z - w * x)],
            [2 * (x * z - w * y),           2 * (y * z + w * x),         1 - 2 * (x ** 2 + y ** 2)]
        ])

    # ---------------------------
    # Euler extraction
    # ---------------------------
    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Returns Roll, Pitch, Yaw in degrees (roll, pitch, yaw)."""
        w, x, y, z = self.state

        # Roll (x-axis rotation)
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
