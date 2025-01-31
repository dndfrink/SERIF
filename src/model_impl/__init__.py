
from typing import Any
from abc import ABC, abstractmethod
from numpy import ndarray

class ModelImpl(ABC):
    @abstractmethod
    def forward(self, data : ndarray):
        pass

    @abstractmethod
    def load(self, weights : bytes):
        pass
