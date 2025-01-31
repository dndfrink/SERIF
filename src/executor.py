import time
import random
from collections import Counter
from multiprocessing.synchronize import Lock
from redis import RedisCluster
from src.data_types import DataModality
from typing import List
from pprint import pprint

from src.system_types import ExecutorNode
from src.get_impl import get_impl_for_model
from src.redis_helpers import decode_numpy
from src.kvcache import KVCache
from src.request import (
    InferenceRequest,
    TerminationRequest,
)

from .request_queue import RequestQueue

from .config_parser import SystemConfig, ApplicationConfig, DataConfig


def theoretical_util(req: InferenceRequest):
    input_probs = [0.0, 1.0] if req.true_label == 1 else [1.0, 0.0]
    return req.assigned_model.expected_conditional_utility(input_probs)


def clear_request_data(
        token: str, modalities: List[DataModality], redisStore: RedisCluster, local_kv_store: KVCache):
    for modality in modalities:
        if local_kv_store is not None:
            local_kv_store.delete(modality.modality, token)
        redisStore.delete(f"{token}-{modality.modality}")


def executor_entrypoint(system_config: SystemConfig, application_config: ApplicationConfig, data_config: DataConfig,
                        input_queue: RequestQueue, node_info: ExecutorNode, console_lock: Lock, process_seed: int, stats: List):
    console_lock.acquire()
    print("executor entry -- execution DISABLED")
    console_lock.release()

    shared_kv_store = RedisCluster(
        startup_nodes=system_config.get_redis_cluster_startup_list()
    )

    local_kv_store = None
    cache_data = data_config.get_cache_data_for_server(node_info)
    if cache_data:
        local_kv_store = KVCache(cache_data)

    random.seed(process_seed)

    total_acc = 0.0
    total_util = 0.0
    total_req = 0.0
    total_violations = 0.0
    total_gpu_time = 0.0
    last_model_name = None

    redis_retrieve_stats = {}
    cache_retrieve_stats = {}
    total_latency = 0.0

    # Loop over all applications
    model_impls = {}
    for app in application_config.applications:
        for profile in app.model_profiles:
            if profile.impl["redis-key"] is not None:
                impl_object = get_impl_for_model(profile)
                weight_bytes = shared_kv_store.get(profile.impl["redis-key"])
                impl_object.load(weight_bytes)
                model_impls[profile.name] = impl_object
            redis_retrieve_stats[profile.modality] = {
                "total-req": 0.0, "average": 0.0}
            cache_retrieve_stats[profile.modality] = {
                "total-req": 0.0, "average": 0.0}

    models = []

    while True:
        req = input_queue.get()
        if isinstance(req, TerminationRequest):
            break

        mod = req.assigned_model
        models.append(mod.name)

        request_bytes = None

        t1 = time.time()
        # Try to access request in shared memory first
        if local_kv_store:
            request_bytes = local_kv_store.get(mod.modality, req.token)
        t2 = time.time()

        # Only go to Redis if we couldn't find the request bytes in shared
        # memory
        if request_bytes is None:
            request_bytes = shared_kv_store.get(f"{req.token}-{mod.modality}")
            t3 = time.time()
            redis_retrieve_stats[mod.modality]["total-req"] += 1.0
            redis_retrieve_stats[mod.modality]["average"] = ((redis_retrieve_stats[mod.modality]["average"] * (
                redis_retrieve_stats[mod.modality]["total-req"] - 1.0)) + (t3 - t2)) / redis_retrieve_stats[mod.modality]["total-req"]
        else:
            cache_retrieve_stats[mod.modality]["total-req"] += 1.0
            cache_retrieve_stats[mod.modality]["average"] = ((cache_retrieve_stats[mod.modality]["average"] * (
                cache_retrieve_stats[mod.modality]["total-req"] - 1.0)) + (t2 - t1)) / cache_retrieve_stats[mod.modality]["total-req"]

        data = decode_numpy(request_bytes)

        if data is None:
            continue

        execution_time = 0
        if mod.name in model_impls:
            start_time = time.time()
            model_impls[mod.name].forward(data)
            end_time = time.time()
            execution_time = end_time - start_time
        else:
            execution_time = mod.get_inference_latency()
            if last_model_name != req.assigned_model.name:
                execution_time += mod.get_context_switch_latency()

            time.sleep(execution_time)

        req.mark_complete()
        total_gpu_time += execution_time
        row = mod.confusion_matrix[req.true_label]
        d = sum(row)
        prob_match = [c / d for c in row]
        if False:  # TODO: check accuracy eventually
            pass
        elif random.random() < prob_match[req.true_label]:
            total_acc += 1.0
            req.predicted_label = req.true_label
        else:
            req.predicted_label = abs(req.true_label - 1)

        # TODO: real utility calculations
        # total_util += mod.true_utility(req.true_label, req.predicted_label)
        total_util += theoretical_util(req)

        total_latency += req.get_duration()

        # SLO met?
        if req.over_latency_target():
            total_violations += 1.0
        total_req += 1.0
        last_model_name = mod.name

        clear_request_data(
            req.token,
            data_config.modalities,
            shared_kv_store,
            local_kv_store)

    with console_lock:
        print("Executor results:")
        print("    Num Requests:    ", str(total_req))
        print("    Average Latency: ", str(total_latency / max(1, total_req)))
        print("    Accuracy:        ", str(total_acc / max(1, total_req)))
        print("    Utility:         ", str(total_util / max(1, total_req)))
        print("    SLO Violation %: ", str(
            total_violations / max(1, total_req)))
        print("    Total GPU Time:  ", str(total_gpu_time))
        print("    Cache Usage %:   ", str(sum(
            cache_retrieve_stats[modality]["total-req"] for modality in cache_retrieve_stats) / max(1, total_req)))
        print("    Cache Retrieval Stats Per Modality:")
        pprint(cache_retrieve_stats)
        print("    Redis Retrieval Stats Per Modality:")
        pprint(redis_retrieve_stats)

        c = Counter(models)
        for value, count in c.most_common():
            print(value, count)

        stats[0] = total_req
        stats[1] = total_latency / total_req
        stats[2] = total_acc / total_req
