import torch
import numpy as np
import torch.nn as nn
from .clip import clip
from .region_awareness import get_backbone

class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        # x shape: [B, 4, 1, 224, 224]
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        x = self.conv(x)

        x = x.view(B, T, -1)

        x = x.mean(dim=1)

        return x

class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 3, kernel_size=5, stride=5
        )  # (1120, 1120) -> (224, 224)
        self.encoder, self.preprocess = clip.load(name, device="cpu")
        self.backbone = get_backbone()
        self.motion_encoder = MotionEncoder()

    def forward(self, x, feature, motion_maps):
        motion_feature = self.motion_encoder(motion_maps)

        feature = torch.cat([feature, motion_feature], dim=1)

        return self.backbone(x, feature)

    def get_features(self, x):
        x = self.conv1(x)
        features = self.encoder.encode_image(x)
        return features


class RALoss(nn.Module):
    def __init__(self):
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        loss = 0.0
        batch_size = alphas_org[0].shape[0]
        for i in range(len(alphas_org)):
            loss_wt = 0.0
            for j in range(batch_size):
                loss_wt += torch.tensor([10.0], device=alphas_max[i][j].device) / torch.exp(
                    alphas_max[i][j] - alphas_org[i][j]
                )
            loss += loss_wt / batch_size
        return loss
 