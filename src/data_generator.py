import random
from typing import List, Dict, Union, Tuple

from src.types import Application
from src.data_types import DataModality, Stream
from .config_parser import DataConfig


class DataGenerator:
    def __init__(self, data_config: DataConfig):
        self._full_x_data = {}
        self._y = -1
        self._modalities: Dict[str, DataModality] = {}
        for modality in data_config.modalities:
            self._modalities[modality.modality] = modality

    def cache_data_for_tasks(
        self,
        apps: List[Application],
        data_division: Dict[str, Union[Tuple, List[Stream]]],
    ):
        self._full_x_data.clear()
        self._y = random.randint(0, 1)
        modalities = self._get_modalities_for_tasks(apps)
        for modality in modalities:
            self._cache_data_for_modality(modality, data_division)

    def read_data_for_task(self, task: Application, group_id: int):
        data = {}
        for modality in task.get_supported_modalities():
            try:
                data[modality] = self._full_x_data[modality][group_id]
            except KeyError:
                continue
        return data, self._y

    def _cache_data_for_modality(
        self,
        modality: str,
        data_division: Dict[str, List[Union[int, Stream]]],
    ):
        modality_obj = self._modalities[modality]
        if modality not in self._full_x_data:
            self._full_x_data[modality] = {}

        if modality_obj.use_synthetic_data:
            for group_id in data_division["synthetic-range"]:
                self._full_x_data[modality][group_id], _ = modality_obj.next()
        else:
            for stream in data_division[modality]:
                self._full_x_data[modality][stream.group_id] = modality_obj.read(
                    stream)

    def _get_modalities_for_tasks(self, apps: List[Application]):
        modalities = set()
        for app in apps:
            modalities = modalities.union(app.get_supported_modalities())
        return modalities
