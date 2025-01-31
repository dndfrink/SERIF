from abc import ABC, abstractmethod
from numpy import ndarray

class DataImpl(ABC):
    @abstractmethod
    def read(self, filename : str) -> ndarray:
        pass