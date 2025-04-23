from scipy.spatial.transform import Rotation
import torch
import numpy as np

path = "/home/mnikolaus/code/data/collected_data/2025_04_02-16_46_15/Panda 102 follower/ee_pos.pt"
t = torch.load(path)

print("quats")
print(t[50][3:])
print(t[51][3:])

print("euler xyz")
print(Rotation.from_quat(t[51][3:]).as_euler("xyz"))
print(Rotation.from_quat(t[50][3:]).as_euler("xyz"))

print(Rotation.from_euler("xyz", Rotation.from_quat(t[51][3:]).as_euler("xyz")).as_euler("xyz"))
print(Rotation.from_euler("xyz", Rotation.from_quat(t[50][3:]).as_euler("xyz")).as_euler("xyz"))

print("rot in quat")
print((Rotation.from_quat(t[51][3:]) * Rotation.from_quat(t[50][3:]).inv()).as_quat())
print(Rotation.from_euler("xyz", (Rotation.from_quat(t[51][3:]) * Rotation.from_quat(t[50][3:]).inv()).as_euler("xyz")).as_quat())

print("rot in euler xyz")
print((Rotation.from_quat(t[51][3:]) * Rotation.from_quat(t[50][3:]).inv()).as_euler("xyz"))



rot_quat = Rotation.from_euler("xyz", Rotation.from_quat(t[51][3:]).as_euler("xyz"))
rot_quat_prev = Rotation.from_euler("xyz", Rotation.from_quat(t[50][3:]).as_euler("xyz"))

print("roat in euler xyz new")
print((rot_quat * rot_quat_prev.inv()).as_quat())