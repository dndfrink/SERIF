from src.types import ModelInstance
from typing import List

def get_fastest_model(mods: List[ModelInstance]):
    return sorted(mods, key=lambda m: m.get_inference_latency())[0]