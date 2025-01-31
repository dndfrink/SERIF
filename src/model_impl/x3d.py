import gc
import numpy as np
import os
import sys
import time
import io

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import pytorchvideo.data
import torch.utils.data
from torchvision.transforms import Compose, Lambda


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip
)

from src.model_impl import ModelImpl

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

my_dev = 'cuda'

# ImageNet mean and deviation
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# Input video fps
frames_per_second = 30

# Select correct transformation params depending on the chosen model
model_transform_params  = {
    "x3d_s": {
        "random_scale_min": 182,
        "random_scale_max": 225,
        "side_size": 182,
        "crop_size": 160,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "random_scale_min": 256,
        "random_scale_max": 320,
        "side_size": 256,
        "crop_size": 224,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "random_scale_min": 362,
        "random_scale_max": 453,
        "side_size": 362,
        "crop_size": 312,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

def make_video_model(num_classes, model_name):
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    model.blocks[5].proj = nn.Linear(2048, num_classes, bias=True)
    return model

class VideoClassificationLightningModule(L.LightningModule):
    """LightningModule for the action classification task"""
    def __init__(self, learning_rate, num_classes, name):
        super().__init__()
        self.model = make_video_model(num_classes, name)
        self.lr = learning_rate
        self.num_classes = num_classes
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"], weight=custom_weights)
        self.train_acc(y_hat, batch['label'])
        return {'loss':loss, 'pred_labels': y_hat, 'actual_labels': batch['label']}

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"], weight=custom_weights)
        self.valid_acc(y_hat, batch['label'])
        return {'loss':loss, 'pred_labels': y_hat, 'actual_labels': batch['label']}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = OneCycleLR(optimizer, max_lr=2e-3, epochs=15, steps_per_epoch=167) # 78800 ?
        return [optimizer], [scheduler]


def load_x3d_model(name : str, weights : bytes):
    device = torch.device('cuda')
    classification_module = VideoClassificationLightningModule(learning_rate = 8e-5, num_classes = 2, name=name).to(device)

    print("Attempting to load " + name)

    buf = io.BytesIO(weights)
    state_dict = torch.load(buf, map_location=torch.device('cuda'))
    classification_module.load_state_dict(state_dict['state_dict'])
    return classification_module

class X3DImpl(ModelImpl):
    def __init__(self, name):
        self.model = None
        self.name = name
        self.dev = 'cuda'
        self.infer_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(model_transform_params[self.name]["num_frames"]),
                    ShortSideScale(size=model_transform_params[self.name]["side_size"]),
                    CenterCrop(model_transform_params[self.name]["crop_size"])
                ]),
            ),
        ])

    def load(self, weights : bytes):
        self.model = load_x3d_model(self.name, weights)
        self.model.eval()

    def forward(self, data : np.ndarray):
        device = torch.device(self.dev)
        self.model.to(device)

        sh = data.shape
        inputs = torch.from_numpy(
            data.reshape((1, sh[0], sh[1], sh[2], sh[3]))
        ).float().to(self.dev)
        output = self.model(inputs).data.cpu()
        _, predicted = torch.max(output, dim=1)

        return predicted

if __name__ == '__main__':
    x = X3DImpl('x3d_l')
    x.load()
    data = np.random.rand(3, 16, 320, 568)
    print(x.forward(data))
