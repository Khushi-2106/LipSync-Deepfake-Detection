import os
import torch
import torch.nn as nn
from models import build_model, get_loss


class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )

        self.model = build_model(opt.arch)
        self.step_bias = 0
            
        if opt.fine_tune and os.path.exists(opt.pretrained_model):
            # state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            # self.model.load_state_dict(state_dict["model"], strict=False)
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            pretrained_dict = state_dict["model"]
            model_dict = self.model.state_dict()

            # Filter out mismatched keys
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping: {k} (shape mismatch)")

            # Load only matching weights
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)

            print(f"Model partially loaded from {opt.pretrained_model}")
            
            self.total_steps = state_dict.get("total_steps", 0)
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")
        else:
            print("Training from scratch")
        if opt.fix_encoder:
            # params = []
            # for name, p in self.model.named_parameters():
            #     if opt.fix_encoder and name.split(".")[0] == "encoder":
            #         p.requires_grad = False
            #     else:
            #         p.requires_grad = True
            #         params.append(p)
            # params = self.model.parameters()
            params = []

            for name, p in self.model.named_parameters():
                if "motion_encoder" in name or "motion_proj" in name or "backbone.fc" in name:
                    p.requires_grad = True
                    params.append(p)
                else:
                    p.requires_grad = False

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.BCEWithLogitsLoss()

        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [
            [t.to(self.device) for t in sublist]
            for sublist in input[1]
        ]
        self.motion_maps = {
            "head": input[2]["head"].to(self.device),
            "face": input[2]["face"].to(self.device),
            "lip":  input[2]["lip"].to(self.device),
        }
        self.label = input[3].to(self.device).float()

    def forward(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features, self.motion_maps
        )
        self.output = self.output.view(-1)
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

        print("Output:", self.output)
        print("Label:", self.label)

        print("INPUT:", self.input.shape)
        print("FEATURE:", self.features.shape)
        print("MOTION:", self.motion_maps["lip"].shape)

    def get_loss(self):
        self.model.to(self.device)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    # def get_features(self):
    #     B, T, C, H, W = self.input.shape
    #     x = self.input.mean(dim=1)

    #     # # Merge batch and time
    #     # x = self.input.view(B * T, C, H, W)

    #     # features = self.model.get_features(x)   # (B*T, 768)

    #     # # Restore shape
    #     # features = features.view(B, T, -1)

    #     # # Aggregate (mean over time)
    #     # self.features = features.mean(dim=1)  # shape: (batch_size

    #     # self.input: [B, T, 3, 1120, 1120]

    #     B, T, C, H, W = self.input.shape

    #     # Take ONE representative frame (or mean)
    #     # x = self.input[:, 0]   # shape → [B, 3, 1120, 1120]

    #     elif self.input.dim() == 4:
    #     # Image input: [B, C, H, W]
    #         x = self.input

    #     else:
    #         raise ValueError(f"Unexpected input shape: {self.input.shape}")

    #     self.features = self.model.get_features(x).to(self.device)

    def get_features(self):

        if self.input.dim() == 5:
            # Video input: [B, T, C, H, W]
            B, T, C, H, W = self.input.shape

            # Option 1: take first frame
            x = self.input[:, 0]

            # Option 2 (better later): mean over time
            # x = self.input.mean(dim=1)

        elif self.input.dim() == 4:
            # Image input: [B, C, H, W]
            x = self.input

        else:
            raise ValueError(f"Unexpected input shape: {self.input.shape}")

        self.features = self.model.get_features(x).to(self.device)

        print("INPUT SHAPE:", self.input.shape)
        print("FEATURE SHAPE:", self.features.shape)

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)
