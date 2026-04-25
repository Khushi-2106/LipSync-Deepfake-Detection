import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils
import numpy as np

def compute_optical_flow(frame1, frame2):
    # Convert tensor (C, H, W) → numpy (H, W, C)
    frame1 = frame1.permute(1, 2, 0).cpu().numpy()
    frame2 = frame2.permute(1, 2, 0).cpu().numpy()

    # Convert to uint8 (OpenCV expects this)
    frame1 = frame1.astype(np.uint8)
    frame2 = frame2.astype(np.uint8)

    # Convert to grayscale
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        prev, nxt,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Convert to magnitude (IMPORTANT)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize (VERY IMPORTANT)
    # mag = mag / (np.max(mag) + 1e-6)
    #Global normalization- prevents exaggeration
    mag = np.clip(mag / 10.0, 0, 1)

    mag = cv2.GaussianBlur(mag, (5,5), 0)   #reduces noise
    mag[mag < 0.02] = 0   #reduces background noise
    # Convert back to tensor
    mag = torch.tensor(mag, dtype=torch.float32)

    return mag

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
)

class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        import glob

        self.real_list = glob.glob("./datasets/AVLips/0_real/*.png")
        self.fake_list = glob.glob("./datasets/AVLips/1_fake/*.png")

        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1
        self.total_list = self.real_list + self.fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        video_path = img_path
        label = self.label_dict[img_path]
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
    
        # crop images
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        frame_strip = img[:, 500:, :]   # remove spectrogram part

        frames = []
        for i in range(5):
            frame = frame_strip[:, :, i*500:(i+1)*500]
            frame = transforms.Resize((224, 224))(frame)
            frames.append(frame)

        import torchvision.transforms as T

        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ])
        frame = self.transform(frame)

        crops = [frames, [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(normalize(transforms.Resize((224, 224))(
                crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]
            )))
            crops[2].append(normalize(transforms.Resize((224, 224))(
                crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]
            )))
        # motion_maps = []

        # for i in range(len(crops[0]) - 1):
        #     flow_map = compute_optical_flow(crops[0][i], crops[0][i+1])
        #     motion_maps.append(flow_map)

        # motion_maps = torch.stack(motion_maps)  # (T-1, H, W)

        #--------------------------
        head_flows = []
        face_flows = []
        lip_flows = []

        for i in range(len(crops[0]) - 1):
            head_flow = compute_optical_flow(crops[0][i], crops[0][i+1])
            face_flow = compute_optical_flow(crops[1][i], crops[1][i+1])
            lip_flow  = compute_optical_flow(crops[2][i], crops[2][i+1])

            head_flows.append(head_flow)
            face_flows.append(face_flow)
            lip_flows.append(lip_flow)

        # Stack them
        head_flows = torch.stack(head_flows)
        face_flows = torch.stack(face_flows)
        lip_flows = torch.stack(lip_flows)
        #-----------------------------------
        
        img = transforms.Resize((1120, 1120))(img)

        head_flows = head_flows.unsqueeze(1)   # (T-1, 1, H, W)
        face_flows = face_flows.unsqueeze(1)
        lip_flows  = lip_flows.unsqueeze(1)

        if idx == 0:
            print("IMG shape:", img.shape)
            print("Number of frames:", len(crops[0]))
            print("Single crop shape:", crops[0][0].shape)
            #print("Motion maps shape:", motion_maps.shape)
            #print("Motion map min/max:", motion_maps.min().item(), motion_maps.max().item())

            # import matplotlib.pyplot as plt

            # for i in range(5):
            #     frame_np = frames[i].permute(1,2,0).cpu().numpy().astype(np.uint8)
            #     plt.imshow(frame_np)
            #     plt.title(f"Frame {i}")
            #     plt.show()

            # import matplotlib.pyplot as plt

            # for i in range(len(motion_maps)):
            #     plt.imshow(motion_maps[i].cpu().numpy(), cmap='jet')
            #     plt.title(f"Flow {i}")
            #     plt.colorbar()
            #     plt.show()

        # return img, crops, motion_maps, label, video_path

        motion_maps = {
            "head": head_flows,
            "face": face_flows,
            "lip": lip_flows
        }
        return img, crops, motion_maps, label, video_path
