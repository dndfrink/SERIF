from multiprocessing import shared_memory
from typing import Dict
from datetime import datetime, timezone
from fasteners import InterProcessReaderWriterLock
import numpy as np
import pickle


class KVCache:
    def __init__(self, modality_map: Dict[str, Dict[str, int]]):
        self.lock = InterProcessReaderWriterLock("/tmp/kvcache.lock")

        # TODO: calculate size of each entry rather than hacking it in via a field
        # Try to connect to existing kv_store, create if doesn't exist
        try:
            self.kv_store = shared_memory.SharedMemory(name='kvcache')
        except FileNotFoundError:
            # Calculate storage size
            storage_in_bytes = 0
            num_total_entries = 0
            for modality in modality_map:
                storage_in_bytes += (modality_map[modality]
                                     ["size"]) * modality_map[modality]["entries"]
                num_total_entries += modality_map[modality]["entries"]
            print(f"Allocating {storage_in_bytes} bytes for data.")
            self.kv_store = shared_memory.SharedMemory(
                name='kvcache', create=True, size=storage_in_bytes)

        # Try to connect to existing metadata, create if doesn't exist
        try:
            self.metadata = shared_memory.SharedMemory(name='kvcache_metadata')
            sorted_keys = sorted(modality_map.keys())
            entry_index = 0
            self.indices = {}
            for key in sorted_keys:
                self.indices[key] = [entry_index,
                                     entry_index + modality_map[key]["entries"]]
                entry_index += modality_map[key]["entries"]
        except FileNotFoundError:
            metadata_template = {}
            sorted_keys = sorted(modality_map.keys())
            entry_index = 0
            self.indices = {}
            for key in sorted_keys:
                self.indices[key] = [entry_index,
                                     entry_index + modality_map[key]["entries"]]
                for _ in range(modality_map[key]["entries"]):
                    metadata_template[entry_index] = {"inuse": 0, "timestamp": "placeholder111",
                                                      "offset": "placeholder222", "size": "placeholder333",
                                                      "key": "0" * 30}
                    entry_index += 1

            max_metadata_len = len(pickle.dumps(metadata_template))

            # Since it doesn't take *that* much space to store the metadata, just
            # allocate quadruple the estimated space and we definitely won't
            # run into any issues
            shared_mem_metadata_len = max_metadata_len * 4
            print(f"Allocating {shared_mem_metadata_len} bytes for metadata")
            self.metadata = shared_memory.SharedMemory(
                name='kvcache_metadata', create=True, size=shared_mem_metadata_len * 2)
            # Initialize metadata only if we just created it
            offset = 0
            entry_index = 0
            for key in sorted_keys:
                modality_size = modality_map[key]["size"]
                for _ in range(modality_map[key]["entries"]):
                    metadata_template[entry_index]["timestamp"] = self.get_unix_time(
                    )
                    metadata_template[entry_index]["size"] = 0
                    metadata_template[entry_index]["offset"] = offset
                    offset += modality_size
                    entry_index += 1

            with self.lock.write_lock():
                self.write_metadata_locked(metadata_template)

    # NOTE: The calling function should have a write lock acquired when
    # calling this
    def write_metadata_locked(self, new_metadata: Dict) -> None:
        metadata_bytes = pickle.dumps(new_metadata)
        metadata_len = len(metadata_bytes)
        metadata_arr_wrapper = np.ndarray(
            (1,), dtype=np.int32, buffer=self.metadata.buf[0:4])
        metadata_arr_wrapper[0] = metadata_len
        self.metadata.buf[4:4 + metadata_len] = metadata_bytes

    def get_unix_time(self) -> int:
        return round((datetime.now(timezone.utc) - datetime(year=1970,
                     month=1, day=1, tzinfo=timezone.utc)).total_seconds())

    # NOTE: The calling function should have at least a read lock acquired
    # when calling this
    def read_metadata(self) -> Dict:
        return pickle.loads(self.metadata.buf[4:])

    def find_slot_for_modality_key(
            self, metadata: Dict, modality: str, key: str) -> int:
        slot_range = self.indices[modality]
        # first, check if the key already exists
        for i in range(slot_range[0], slot_range[1]):
            if metadata[i]["inuse"] == 1 and metadata[i]["key"] == key:
                return i
        # if key does not exist, find an empty slot
        for i in range(slot_range[0], slot_range[1]):
            if metadata[i]["inuse"] == 0:
                return i
        return -1

    def set(self, modality: str, key: str, value: bytes) -> bool:
        with self.lock.write_lock():
            metadata = self.read_metadata()
            slot = self.find_slot_for_modality_key(metadata, modality, key)
            if slot >= 0:
                # slot is available, set new metadata values
                metadata[slot]["inuse"] = 1
                metadata[slot]["timestamp"] = self.get_unix_time()
                metadata[slot]["size"] = len(value)
                metadata[slot]["key"] = key

                self.write_metadata_locked(metadata)

                offset = metadata[slot]["offset"]
                self.kv_store.buf[offset:offset + len(value)] = value
            return slot >= 0

    def get(self, modality: str, key: str) -> bytes:
        with self.lock.read_lock():
            metadata = self.read_metadata()
            slot_range = self.indices[modality]
            for i in range(slot_range[0], slot_range[1]):
                if metadata[i]["inuse"] == 1 and metadata[i]["key"] == key:
                    offset = metadata[i]["offset"]
                    return bytes(
                        self.kv_store.buf[offset:offset + metadata[i]["size"]])
            return None

    def delete(self, modality: str, key: str) -> bool:
        with self.lock.write_lock():
            metadata = self.read_metadata()
            slot_range = self.indices[modality]
            for i in range(slot_range[0], slot_range[1]):
                if metadata[i]["inuse"] == 1 and metadata[i]["key"] == key:
                    metadata[i]["inuse"] = 0
                    self.write_metadata_locked(metadata)
                    return True
            return False

    def purge_latent_entries(self) -> None:
        with self.lock.write_lock():
            current_timestamp = self.get_unix_time()
            metadata = self.read_metadata()
            for key in metadata.keys():
                if (current_timestamp - metadata[key]["timestamp"] > 15):
                    metadata[key]["inuse"] = 0
            self.write_metadata_locked(metadata)
