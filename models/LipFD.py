import torch
import numpy as np
import torch.nn as nn
from .clip import clip
from .region_awareness import get_backbone

class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()

        def make_branch():
            return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head_branch = make_branch()
        self.face_branch = make_branch()
        self.lip_branch  = make_branch()

        self.attn_fc = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 weights: head, face, lip
            nn.Softmax(dim=1)
        )

    def forward(self, motion_maps):

        def process(flow, branch):
            B, T, C, H, W = flow.shape
            flow = flow.view(B*T, C, H, W)
            feat = branch(flow)
            feat = feat.view(B, T, -1)
            return feat.mean(dim=1)  # [B, 64]

        head_feat = process(motion_maps["head"], self.head_branch)
        face_feat = process(motion_maps["face"], self.face_branch)
        lip_feat  = process(motion_maps["lip"],  self.lip_branch)

        # Stack features
        feats = torch.stack([head_feat, face_feat, lip_feat], dim=1)  # [B, 3, 64]

        # Compute attention weights
        attn_input = torch.cat([head_feat, face_feat, lip_feat], dim=1)  # [B, 192]
        weights = self.attn_fc(attn_input)  # [B, 3]

        # Apply attention
        weights = weights.unsqueeze(-1)  # [B, 3, 1]
        weighted_feats = feats * weights

        # Final motion feature
        motion_feature = weighted_feats.sum(dim=1)  # [B, 64]

        return motion_feature

class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 3, kernel_size=5, stride=5
        )  # (1120, 1120) -> (224, 224)
        self.encoder, self.preprocess = clip.load(name, device="cpu")
        self.backbone = get_backbone()
        self.dropout = nn.Dropout(p=0.5)
        self.motion_encoder = MotionEncoder()
        self.motion_proj = nn.Linear(64, 256)

    def forward(self, crops, feature, motion_maps):

        # motion feature
        motion_feature = self.motion_encoder(motion_maps)
        motion_feature = self.motion_proj(motion_feature)

        # combine
        feature = torch.cat([feature, motion_feature], dim=1)
        feature = self.dropout(feature)
        
        print("CROPS LENGTH:", len(crops))
        print("FRAMES PER REGION:", len(crops[0]))
        print("ONE FRAME SHAPE:", crops[0][0].shape)

        return self.backbone(crops, feature)


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
 