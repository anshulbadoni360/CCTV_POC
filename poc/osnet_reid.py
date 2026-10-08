"""Optional OSNet MSMT17 ReID encoder adapted from the supplied prototype.

Torch, torchvision, and huggingface_hub are imported only when this optional
backend is instantiated. This keeps the base app usable without heavyweight ML
packages. Its embeddings are appearance cues, not biometric identity proof.
"""
from __future__ import annotations

import ssl
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download


OSNET_REID_REPO = "kaiyangzhou/osnet"
OSNET_REID_FILENAME = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
    "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride, padding, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu else x


class Conv1x1(ConvLayer):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__(in_c, out_c, 1, stride, 0, groups)


class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_c, num_gates=None, reduction=16):
        super().__init__()
        num_gates = num_gates or in_c
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, num_gates, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=4):
        super().__init__()
        mid = out_c // reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1(mid, out_c)
        self.downsample = Conv1x1(in_c, out_c) if in_c != out_c else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(self, feature_dim=512, blocks=(2, 2, 2), channels=(64, 256, 384, 512)):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    @staticmethod
    def _make_layer(n_blocks, in_c, out_c):
        layers = [OSBlock(in_c, out_c)]
        for _ in range(n_blocks - 1):
            layers.append(OSBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        x = self.gap(x).flatten(1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)


class OSNetReID:
    """Batched OSNet feature extractor. Cosine scores need camera-specific calibration."""

    def __init__(self, device: str = "auto", allow_download: bool = False):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if device == "auto" else device))
        weight_path = hf_hub_download(
            repo_id=OSNET_REID_REPO,
            filename=OSNET_REID_FILENAME,
            local_files_only=not allow_download,
        )
        try:
            state = torch.load(weight_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model = OSNet(feature_dim=512)
        model_dict = model.state_dict()
        matched = {}
        for key, value in state.items():
            clean = key.replace("module.", "")
            if clean in model_dict and tuple(value.shape) == tuple(model_dict[clean].shape):
                matched[clean] = value
        if len(matched) < 100:
            raise RuntimeError(f"Only {len(matched)} OSNet checkpoint tensors matched the model.")
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        self.model = model.to(self.device).eval()
        self.backend = "osnet_msmt17"
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch).detach().cpu().numpy().astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features
