import random

from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment

def random_scheduler(inp: SchedulerInput) -> Schedule:
    schedule = { node : [] for node in inp.executors}
    req_keys = list(inp.requests.keys())

    random.shuffle(req_keys)
    for req_key in req_keys:
        schedule[random.choice(inp.executors)].append(
            ModelAssignment(
                job_id=req_key,
                model=random.choice(inp.requests[req_key]),
            )
        )

    return schedule

def get_random_executor_for_cached_request(inp : SchedulerInput, server_ip_with_cached_request : str):
    return random.choice([executor for executor in inp.executors if executor.ip_address == server_ip_with_cached_request])

def random_prioritize_cache(inp: SchedulerInput) -> Schedule:
    schedule = { node : [] for node in inp.executors}
    req_keys = list(inp.requests.keys())
    not_cached = []
    for req_key in req_keys:
        if len(inp.kv_cache_info[req_key]) > 0:
            cached_modalities = inp.kv_cache_info[req_key]
            modality_to_use = random.choice(list(cached_modalities.keys()))
            schedule[get_random_executor_for_cached_request(inp, cached_modalities[modality_to_use])].append(
                ModelAssignment(job_id=req_key,
                                model=random.choice([model for model in inp.requests[req_key] if model.modality == modality_to_use])
                )
            )
        else:
            not_cached.append(req_key)
    
    for req_key in not_cached:
        schedule[random.choice(inp.executors)].append(
            ModelAssignment(
                job_id=req_key,
                model=random.choice(inp.requests[req_key]),
            )
        )

    return schedule


RandomPrioritizeCacheScheduler : Scheduler = random_prioritize_cache
RandomScheduler: Scheduler = random_scheduler