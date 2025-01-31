import copy
import random

from typing import List, Tuple, Dict

from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment, evaluate_schedule
from src.system_types import ExecutorNode
from src.types import ModelInstance

def copy_dict(d : Dict):
    return {k: copy.deepcopy(v) for k, v in d.items()}

def get_cached_modalities_for_executor(modalities_cached :  Dict, executor : ExecutorNode):
    modalities = []
    for modality, server_ip in modalities_cached.items():
        if server_ip == executor.ip_address:
            modalities.append(modality)
    return modalities

def locally_optimal_instance(
    inp: SchedulerInput, 
    current_schedule: Schedule,
    request_token: str,
    model_list: List[ModelInstance],
    deadline: float,
    force_cache: bool = False
) -> Tuple[int, ModelInstance]:
    
    executors = []
    models = set()

    if len(inp.kv_cache_info[request_token]) > 0 and force_cache:
        for executor in inp.executors:
            cached_modalities = get_cached_modalities_for_executor(inp.kv_cache_info[request_token], executor)
            if len(cached_modalities) > 0:
                executors.append(executor)
                for model in model_list:
                    if model.modality in cached_modalities:
                        models.add(model)
        models = list(models)
    else:
        executors = inp.executors
        models = model_list

    candidates = [
        (executor, mod)
        for executor in executors
        for mod in models
    ]

    best_candidate = None
    best_utility = -1000000

    for candidate in candidates:
        tmp_sched = copy_dict(current_schedule)
        tmp_sched[candidate[0]].append(
            ModelAssignment(job_id=request_token, model=candidate[1])
        )

        exp_util, violation = evaluate_schedule(
            tmp_sched, inp.data_cache, inp.use_data_awareness,
            inp.deadlines, inp.last_schedule, inp.executor_delay,
            inp.retrieval_latencies, inp.kv_cache_info 
        )

        # I think we determined all candidates are feasible above...
        prefer_this_schedule = True if exp_util > best_utility else False

        if best_candidate is None or prefer_this_schedule:
            best_candidate = candidate
            best_utility = exp_util

    return best_candidate


def locally_optimal_edf_scheduler(inp: SchedulerInput) -> Schedule:
    schedule = { node : [] for node in inp.executors}
    requests = inp.get_sorted_requests_by_deadline()

    for req in requests:
        token = req[0]
        mod_list = req[1]
        executor, mod = locally_optimal_instance(
            inp, schedule, token, mod_list, inp.deadlines[token]
        )
        schedule[executor].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

LocallyOptimalScheduler: Scheduler = locally_optimal_edf_scheduler

def locally_optimal_edf_scheduler_cache(inp: SchedulerInput) -> Schedule:
    schedule = { node : [] for node in inp.executors}
    requests = inp.get_sorted_requests_by_deadline()
    for req in requests:
        token = req[0]
        mod_list = req[1]
        executor, mod = locally_optimal_instance(
            inp, schedule, token, mod_list, inp.deadlines[token], True
        )
        schedule[executor].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

LocallyOptimalPrioritizeCacheScheduler : Scheduler = locally_optimal_edf_scheduler_cache

def locally_optimal_fcfs_scheduler(inp: SchedulerInput) -> Schedule:
    schedule = { i:[] for i in range(inp.num_executors)}
    requests = inp.requests.items()
    #requests = inp.get_sorted_requests_by_deadline()

    for req in requests:
        token = req[0]
        mod_list = req[1]
        executor_idx, mod = locally_optimal_instance(
            inp, schedule, token, mod_list, inp.deadlines[token]
        )
        schedule[executor_idx].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

LocallyOptimalFCFSScheduler: Scheduler = locally_optimal_fcfs_scheduler


def locally_optimal_priority_scheduler(inp: SchedulerInput) -> Schedule:
    """
    Orders requests by priority rather than EDF.
    We could do this by variance I guess, but I think the range is probably
    more important than average distance from the mean utility?
    """
    schedule = { i:[] for i in range(inp.num_executors)}
    requests = inp.get_sorted_requests_by_priority()

    for req in requests:
        token = req[0]
        mod_list = req[1]
        worker_id, mod = locally_optimal_instance(
            inp, schedule, token, mod_list, inp.deadlines[token]
        )
        schedule[worker_id].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

LocallyOptimalPriorityScheduler: Scheduler = locally_optimal_priority_scheduler


def _get_max_util_model_for_request(inp, req_key, mod_list):
    highest_utility = -100000.0
    highest_utility_mod = None
    for mod in mod_list:
        mod_util = inp.get_model_utility(req_key, mod)

        if mod_util > highest_utility or highest_utility_mod is None:
            highest_utility = mod_util
            highest_utility_mod = mod

    return highest_utility_mod

def locally_optimal_priority_max_scheduler(inp: SchedulerInput) -> Schedule:
    """
    Orders requests by priority rather than EDF.
    We could do this by variance I guess, but I think the range is probably
    more important than average distance from the mean utility?
    """
    schedule = { i:[] for i in range(inp.num_executors)}
    requests = inp.get_sorted_requests_by_priority()

    for req in requests:
        token = req[0]
        mod_list = req[1]
        mod = _get_max_util_model_for_request(inp, token, mod_list)

        # TODO HACK for a single worker
        schedule[0].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

LocallyOptimalPriorityMaxScheduler: Scheduler = locally_optimal_priority_scheduler