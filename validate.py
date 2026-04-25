import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score


# def validate(model, loader, gpu_id):
#     print("validating...")
#     device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
#     with torch.no_grad():
#         y_true, y_pred = [], []
#         video_results = {}
#         for img, crops, motion_maps, label, video_path in loader:
#             img_tens = img.to(device)
#             motion_maps = {
#                 "head": motion_maps["head"].to(device),
#                 "face": motion_maps["face"].to(device),
#                 "lip": motion_maps["lip"].to(device),
#             }
#             crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
#             # features = model.get_features(img_tens).to(device)
#             # use same logic as trainer.get_features()
#             x = img[:, 0] if len(img.shape) == 5 else img
#             features = model.get_features(x)

#             output, _, _ = model.forward(crops_tens, features, motion_maps)

#             pred = model(crops_tens, features, motion_maps)[0].sigmoid().flatten().tolist()


#             for p, vp in zip(pred, video_path):
#                 label_name = "FAKE" if p >= 0.5 else "REAL"
#                 print(f"{vp} → {label_name} ({p:.3f})")

#                 base_name = vp.split("/")[-1].rsplit("_", 1)[0]

#                 if base_name not in video_results:
#                     video_results[base_name] = []

#                 video_results[base_name].append(p)

#             y_pred.extend(pred)
#             y_true.extend(label.flatten().tolist())
            
#         print("\nFinal Video-Level Results:")

#         for video, scores in video_results.items():
#             avg_score = sum(scores) / len(scores)
#             final_label = "FAKE" if avg_score >= 0.5 else "REAL"
#             print(f"{video} → {final_label} (avg score = {avg_score:.3f})")
#     y_true = np.array(y_true)
#     y_scores = np.array(y_pred)

#     ap = average_precision_score(y_true, y_scores)

#     y_pred = np.where(y_scores >= 0.5, 1, 0)
#     cm = confusion_matrix(y_true, y_pred)
#     tp, fn, fp, tn = cm.ravel()
#     fnr = fn / (fn + tp)
#     fpr = fp / (fp + tn)
#     acc = accuracy_score(y_true, y_pred)
#     return ap, fpr, fnr, acc

def validate(model, loader, gpu_id):
    print("validating...")

    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    y_true, y_pred = [], []
    video_results = {}

    with torch.no_grad():
        for img, crops, motion_maps, label, video_path in loader:

            # =========================
            # MOVE EVERYTHING TO GPU
            # =========================
            img = img.to(device)

            crops = [
                [frame.to(device) for frame in region]
                for region in crops
            ]

            motion_maps = {
                "head": motion_maps["head"].to(device),
                "face": motion_maps["face"].to(device),
                "lip":  motion_maps["lip"].to(device),
            }

            label = label.to(device).float()

            # =========================
            # FIX INPUT SHAPE
            # =========================
            if len(img.shape) == 5:
                x = img[:, 0]   # [B, C, H, W]
            else:
                x = img

            # =========================
            # FEATURE EXTRACTION
            # =========================
            features = model.get_features(x).to(device)

            # =========================
            # SINGLE FORWARD PASS
            # =========================
            output, _, _ = model.forward(crops, features, motion_maps)

            pred = torch.sigmoid(output).view(-1)

            # =========================
            # STORE RESULTS
            # =========================
            for p, vp in zip(pred, video_path):
                p_val = p.item()

                label_name = "FAKE" if p_val >= 0.5 else "REAL"
                print(f"{vp} → {label_name} ({p_val:.3f})")

                base_name = vp.split("/")[-1].rsplit("_", 1)[0]

                if base_name not in video_results:
                    video_results[base_name] = []

                video_results[base_name].append(p_val)

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(label.cpu().numpy())

    print("\nFinal Video-Level Results:")

    for video, scores in video_results.items():
        avg_score = sum(scores) / len(scores)
        final_label = "FAKE" if avg_score >= 0.5 else "REAL"
        print(f"{video} → {final_label} (avg score = {avg_score:.3f})")

    # =========================
    # METRICS
    # =========================
    y_true = np.array(y_true)
    y_scores = np.array(y_pred)

    ap = average_precision_score(y_true, y_scores)

    y_bin = (y_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    fnr = fn / (fn + tp + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    acc = accuracy_score(y_true, y_bin)

    return ap, fpr, fnr, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_list_path", type=str, default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", type=str, default="./datasets/val/1_fake")
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt.pth")
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    
    checkpoint = state_dict["model"]
    model_dict = model.state_dict()

    # Filter only matching keys
    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v

    # Load safely
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(filtered_dict)} / {len(model_dict)} layers from checkpoint")

    print("Model loaded.")
    model.eval()
    model.to(device)

    dataset = AVLip(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )
    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")

