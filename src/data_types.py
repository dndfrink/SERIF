from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from numpy.random import rand, randint
from src.get_impl import get_reader_for_modality

# This class represents a single data modality as represented in the SystemConfig JSON structure.
# A "modality" encompasses the following features of a specific data format:
#   - data type
#   - data shape
#   - input streams for retrieving data from
#   - interfaces to retrieve either real or synthetic data

@dataclass
class Stream:
    stream_id: str
    group_id: int

    def __hash__(self):
        return hash((self.group_id, self.stream_id))

@dataclass
class DataModality:
    modality: str
    shape: List[int]
    use_synthetic_data: bool
    cache_entry_size: int
    retrieval_latencies: Dict[str, float]

    def read(self, stream: Stream) -> np.ndarray:
        reader = get_reader_for_modality(self.modality)
        return reader.read(stream.stream_id)

    def next(self):
        return rand(*self.shape), randint(0, 1)
    
@dataclass
class RequestGenerationNode:
    name: str
    ip_address: str
    port: str
    num_workers: int
    streams: Dict[str, Dict[str, str]]
    group_range: List[str]
    cache_entries_per_modality: Dict[str, int]

    def __init__(self, name: str, ip_address: str, port: str, num_workers: int,
                 streams: Dict[str, Dict[str, str]], cache_entries_per_modality: Dict[str, int]):
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.num_workers = num_workers
        self.streams = streams
        self.group_range = None
        self.cache_entries_per_modality = cache_entries_per_modality

    def __hash__(self):
        return hash((self.name, self.ip_address,
                    self.port, "RequestGenerator"))

    def get_stream(self, modality: str, group_id: str) -> Stream:
        try:
            return Stream(self.streams[modality][group_id], modality)
        except KeyError:
            return Stream(None, modality)

    def get_num_entries_for_modality(self, modality: str) -> int:
        if modality in self.cache_entries_per_modality:
            return self.cache_entries_per_modality[modality]
        else:
            return self.cache_entries_per_modality["default"]

    def get_worker_divisions(
            self) -> List[Dict[str, List[Union[int, Stream]]]]:
        if self.group_range is None or 0 == len(self.group_range):
            return [set() for _ in range(self.num_workers)]

        # Convert to sorted list for consistent division
        sorted_groups = sorted(int(x) for x in self.group_range)
        worker_groups: List[Dict[str, List[Union[str, Stream]]]] = [
            {} for _ in range(self.num_workers)]

        # Calculate base groups per worker and remainder
        base_count = len(sorted_groups) // self.num_workers
        remainder = len(sorted_groups) % self.num_workers

        start = 0
        for i in range(self.num_workers):
            # Add one extra item to this worker if we still have remainder
            count = base_count + (1 if i < remainder else 0)
            end = start + count

            for modality, streams in self.streams.items():
                worker_groups[i][modality] = [
                    Stream(streams[str(x)], x) for x in sorted_groups[start:end]]
            worker_groups[i]["synthetic-range"] = [x for x in sorted_groups[start:end]]
            start = end

        return worker_groups
