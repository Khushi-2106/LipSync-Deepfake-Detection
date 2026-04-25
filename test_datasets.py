import matplotlib.pyplot as plt
import torch
import os

from data.datasets import AVLip
import options

# Load options
class DummyOpt:
    def __init__(self):
        self.data_label = "train"

opt = DummyOpt()

# Dataset
dataset = AVLip(opt)

# Sample
img, crops, motion_maps, label, path = dataset[0]

print("\n===== BASIC INFO =====")
print("Image shape:", img.shape)
print("Number of frames:", len(crops[0]))
print("Frame shape:", crops[0][0].shape)

print("\n===== MULTI-SCALE FLOW INFO =====")
print("HEAD:", motion_maps["head"].shape)
print("FACE:", motion_maps["face"].shape)
print("LIP :", motion_maps["lip"].shape)

print("\n===== VALUE CHECK =====")
print("HEAD min/max:", motion_maps["head"].min().item(), motion_maps["head"].max().item())
print("FACE min/max:", motion_maps["face"].min().item(), motion_maps["face"].max().item())
print("LIP  min/max:", motion_maps["lip"].min().item(), motion_maps["lip"].max().item())

# ---------------------------
# SAVE FLOW IMAGES
# ---------------------------
os.makedirs("debug_flow", exist_ok=True)

plt.imsave("debug_flow/head_flow.png", motion_maps["head"][0][0].cpu(), cmap="jet")
plt.imsave("debug_flow/face_flow.png", motion_maps["face"][0][0].cpu(), cmap="jet")
plt.imsave("debug_flow/lip_flow.png", motion_maps["lip"][0][0].cpu(), cmap="jet")

print("\nSaved flow images in debug_flow/")

print("HEAD mean:", motion_maps["head"].mean().item())
print("FACE mean:", motion_maps["face"].mean().item())
print("LIP mean:", motion_maps["lip"].mean().item())

