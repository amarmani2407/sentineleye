"""Shared code for the learned anomaly detector (Step 6).

A small autoencoder learns to rebuild pictures of your NORMAL scene.
Later, if a frame is rebuilt badly (high error), something unusual is in view.
"""
import os
import json

import cv2
import numpy as np
import torch
from torch import nn

MODEL_FILE = "anomaly_model.pt"
THRESHOLD_FILE = "anomaly_threshold.json"
IMG_W, IMG_H = 64, 48          # every frame is shrunk to this size
TOPK_FRACTION = 0.02           # score = average of the worst 2% of pixels


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder squeezes the picture down to a tiny summary...
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 8, 1))
        # ...and the decoder tries to rebuild the picture from that summary.
        self.dec = nn.Sequential(
            nn.Conv2d(8, 64, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def preprocess(frame):
    """Camera frame (BGR) -> small grey picture with values between 0 and 1."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def batch_scores(model, X):
    """X has shape (N, H, W). Returns one anomaly score per picture."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).unsqueeze(1)
            err = ((model(x) - x) ** 2).flatten(1)
            k = max(1, int(err.shape[1] * TOPK_FRACTION))
            out.append(torch.topk(err, k, dim=1).values.mean(1).numpy())
    return np.concatenate(out)


def available():
    return os.path.exists(MODEL_FILE) and os.path.exists(THRESHOLD_FILE)


class AnomalyDetector:
    def __init__(self):
        self.model = AE()
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
        self.model.eval()
        with open(THRESHOLD_FILE) as f:
            self.threshold = float(json.load(f)["threshold"])

    def score(self, frame):
        x = preprocess(frame)[None]          # shape (1, H, W)
        return float(batch_scores(self.model, x)[0])
