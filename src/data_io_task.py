import time
import random
from typing import Dict, List, Union
from multiprocessing.synchronize import Lock

import numpy as np
from redis import RedisCluster
from xmlrpc.client import ServerProxy
from sneakpeek import SneakPeekModel
from src.types import Application
from src.data_types import Stream, RequestGenerationNode

from src.config_parser import SystemConfig, ApplicationConfig, DataConfig
from src.kvcache import KVCache
from .data_generator import DataGenerator
from .redis_helpers import encode_numpy
from .request import InferenceRequest


def create_and_cache_request(
    app: Application,
    generator: DataGenerator,
    group_id: int,
    shared_kv_store: RedisCluster,
    local_kv_cache: KVCache,
    data_aware: bool,
    sp_models: List[SneakPeekModel],
    requests: List[InferenceRequest],
    server_obj: RequestGenerationNode,
):
    true_label = 1 if random.random() <= 0.1 else 0
    next_request = InferenceRequest(app)
    x_data, y = generator.read_data_for_task(app, group_id)
    next_request.true_label = true_label

    for modality_key in x_data:
        request_data = encode_numpy(x_data[modality_key])
        shared_kv_store.set(
            f"{next_request.token}-{modality_key}",
            request_data,
        )
        if local_kv_cache.set(modality_key, next_request.token, request_data):
            next_request.set_modality_cached(
                modality_key, server_obj.ip_address)

    if data_aware:
        sp_models[app.name].fit(x_data, y)
        next_request.class_probs = sp_models[app.name].infer(
            x_data, next_request.true_label
        )

    requests.append(next_request)


def io_task_entrypoint(
    system_config: SystemConfig,
    application_config: ApplicationConfig,
    data_config: DataConfig,
    server_obj: RequestGenerationNode,
    console_lock: Lock,
    data_division: Dict[str, List[Union[str, Stream]]],
    process_seed: int,
) -> None:
    np.random.seed(process_seed)
    random.seed(process_seed)
    data_aware = system_config.data_aware
    scheduler_proxy = ServerProxy(system_config.scheduler_node.url, allow_none=True)
    possible_apps = application_config.applications
    sp_models: Dict[str, SneakPeekModel] = {}

    for app in possible_apps:
        sp_models[app.name] = SneakPeekModel(app)

    shared_kv_store = RedisCluster(startup_nodes=system_config.get_redis_cluster_startup_list()
)
    local_kv_cache = KVCache(data_config.get_cache_data_for_server(server_obj))

    sys_start_time = time.time()
    sys_end_time = sys_start_time + system_config.duration_in_secs
    run_forever = system_config.duration_in_secs == 0
    generator = DataGenerator(data_config)
    quantum = 1.0

    current_period = 1
    current_time = time.time()

    while current_time < sys_end_time or run_forever:
        required_apps: List[Application] = []
        for app in possible_apps:
            if (current_period % app.get_sample_duration()) == 0:
                required_apps.append(app)

        generator.cache_data_for_tasks(required_apps, data_division)
        requests: List[InferenceRequest] = []

        for app in required_apps:
            for modality in app.get_supported_modalities():
                if modality not in data_division:
                    for group in data_division["synthetic-range"]:
                        create_and_cache_request(
                            app,
                            generator,
                            group,
                            shared_kv_store,
                            local_kv_cache,
                            data_aware,
                            sp_models,
                            requests,
                            server_obj,
                        )
                else:
                    for stream in data_division[modality]:
                        create_and_cache_request(
                            app,
                            generator,
                            stream.group_id,
                            shared_kv_store,
                            local_kv_cache,
                            data_aware,
                            sp_models,
                            requests,
                            server_obj,
                        )

        # Push all jobs to scheduler
        for req in requests:
            req.start_the_clock()
            scheduler_proxy.enqueue(req)

        # Only sleep the required remaining duration
        data_time = time.time() - current_time
        sleep_time = 0 if data_time > quantum else (quantum - data_time)
        time.sleep(sleep_time)
        current_time = time.time()
        current_period += 1

    duration = time.time() - sys_start_time
    with console_lock:
        print(f"Actual duration: {duration}")


def kv_garbage_collector_entrypoint(
    data_config: DataConfig,
    server: RequestGenerationNode,
):
    local_kv_store = KVCache(
        data_config.get_cache_data_for_server(server)
    )
    while True:
        local_kv_store.purge_latent_entries()
        time.sleep(2)
