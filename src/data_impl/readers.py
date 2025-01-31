from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from torch import Tensor
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

from src.data_impl import DataImpl

class VideoReader(DataImpl):
    def __init__(self):
        pass
    
    def read(self, filename: str) -> np.ndarray:
        # ImageNet mean and deviation
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        generatorTransform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std)
                ]),
            ),
        ])

        vid = EncodedVideo.from_path(filename, decode_audio=False)
        vid = vid.get_clip(start_sec=0.0, end_sec=80.0/30.0)
        vid = generatorTransform(vid)
        npArray = vid['video'].numpy().astype(np.float64)
        return npArray
