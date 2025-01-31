from src.scheduler_types import Scheduler
from sneakpeek.scheduling.locally_optimal import LocallyOptimalScheduler, LocallyOptimalPrioritizeCacheScheduler
from sneakpeek.scheduling.cluster import ClusterPriorityDynamicScheduler
from sneakpeek.scheduling.random import RandomScheduler, RandomPrioritizeCacheScheduler
from src.types import ModelInstance
from src.model_impl import ModelImpl
from src.model_impl.x3d import X3DImpl
from src.data_impl.readers import VideoReader
from src.data_impl import DataImpl

def get_reader_for_modality(modality: str) -> DataImpl:
    impl_object: DataImpl = None
    if modality == "video":
        impl_object = VideoReader()
    else:
        raise TypeError("Bad modality")

    # Enforce that the object being returned is correctly derived
    assert issubclass(type(impl_object), DataImpl)
    return impl_object


def get_impl_for_model(mod: ModelInstance) -> ModelImpl:
    impl_object: ModelImpl = None
    model_name = mod.name
    if model_name == "x3d_m" or model_name == "x3d_l":
        impl_object = X3DImpl(model_name)
    else:
        raise TypeError("Bad model type")

    # Enforce that the object being returned is correctly derived
    assert issubclass(type(impl_object), ModelImpl)
    return impl_object


def get_schedule_impl(scheduler_type: str) -> Scheduler:
    if scheduler_type == "random":
        return RandomScheduler
    elif scheduler_type == "random-cache":
        return RandomPrioritizeCacheScheduler
    elif scheduler_type == "grouped-priority":
        return ClusterPriorityDynamicScheduler
    elif scheduler_type == "locally-optimal-edf":
        return LocallyOptimalScheduler
    elif scheduler_type == "locally-optimal-edf-cache":
        return LocallyOptimalPrioritizeCacheScheduler
    else:
        raise TypeError("Bad scheduler string")
