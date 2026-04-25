import torch
from data.datasets import AVLip
import options
from models import build_model

# Load options
class DummyOpt:
    def __init__(self):
        self.data_label = "train"
        self.arch = "CLIP:ViT-L/14"

opt = DummyOpt()

# Dataset
dataset = AVLip(opt)

# Get sample
img, crops, motion_maps, label, _ = dataset[0]

# Add batch dimension
img = img.unsqueeze(0)

motion_maps = {
    "head": motion_maps["head"].unsqueeze(0),
    "face": motion_maps["face"].unsqueeze(0),
    "lip": motion_maps["lip"].unsqueeze(0),
}

crops = [[frame.unsqueeze(0) for frame in sublist] for sublist in crops]

# Load model
model = build_model(opt.arch)
model.eval()

# Forward pass
with torch.no_grad():
    features = model.get_features(img)

    print("\n===== FEATURE CHECK =====")
    print("CLIP feature shape:", features.shape)

    output, w_max, w_org = model.forward(crops, features, motion_maps)

    print("\n===== OUTPUT CHECK =====")
    print("Output:", output)
    print("Output shape:", output.shape)