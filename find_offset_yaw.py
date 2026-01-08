import numpy as np
import pandas as pd

data = pd.read_csv("Data_contest_2/train.csv")

def yaw_from_mag(acc, mag):
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)

    roll = np.arctan2(acc[1], acc[2])
    pitch = np.arcsin(-acc[0])

    mx, my, mz = mag

    mx2 = mx*np.cos(pitch) + mz*np.sin(pitch)
    my2 = mx*np.sin(roll)*np.sin(pitch) + my*np.cos(roll) - mz*np.sin(roll)*np.cos(pitch)

    return np.arctan2(-my2, mx2)

yaw_diffs = []

for _, row in data.iterrows():
    acc = row[['AccX','AccY','AccZ']].values
    mag = row[['MagX','MagY','MagZ']].values

    yaw_mag = yaw_from_mag(acc, mag)
    yaw_gt = np.deg2rad(row['yaw'])

    diff = yaw_gt - yaw_mag
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    yaw_diffs.append(diff)

yaw_offset = np.median(yaw_diffs)
print("Yaw offset [rad]:", yaw_offset)
print("Yaw offset [deg]:", np.degrees(yaw_offset))
