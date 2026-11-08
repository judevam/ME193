"""Finger-count CNN (built from scratch in PyTorch) + shared preprocessing.

Three conv layers of different sizes, a pool after each one, then the pooled
features of ALL three stages are concatenated and a linear layer reduces them to
the class scores (trained with cross-entropy loss).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 96  # network input is IMG x IMG, grayscale
CLASSES = ["0", "1", "2", "3", "4", "5", "empty"]  # 0 = fist; classes with no data are skipped in training


def preprocess(frame_bgr, roi):
    """Full camera frame (BGR) -> float32 array (IMG, IMG) in [0, 1].

    The same function must be used when training and on the UNO Q.
    """
    x, y, s = roi["x"], roi["y"], roi["size"]
    crop = frame_bgr[y:y + s, x:x + s]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


class FingerNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)    # coarse edges
        self.c2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)   # finger-like shapes
        self.c3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # combinations
        self.b1, self.b2, self.b3 = nn.BatchNorm2d(8), nn.BatchNorm2d(16), nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        # each stage is average-pooled to 4x4 so all three can be concatenated
        self.fc = nn.Linear((8 + 16 + 32) * 4 * 4, n_classes)

    def forward(self, x, return_maps=False):
        a1 = self.pool(F.relu(self.b1(self.c1(x))))    # 8  x 48 x 48
        a2 = self.pool(F.relu(self.b2(self.c2(a1))))   # 16 x 24 x 24
        a3 = self.pool(F.relu(self.b3(self.c3(a2))))   # 32 x 12 x 12
        feats = [F.adaptive_avg_pool2d(a, 4).flatten(1) for a in (a1, a2, a3)]
        out = self.fc(self.drop(torch.cat(feats, dim=1)))
        return (out, (a1, a2, a3)) if return_maps else out
