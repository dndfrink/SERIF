import copy
import random
import statistics

from typing import Dict, List, Optional, Tuple

from src.scheduler_types import penalized_util, Schedule, Scheduler, SchedulerInput, ModelAssignment 
from src.types import ModelInstance
from sneakpeek.scheduling import get_fastest_model

def get_fastest_edf_schedule(
    inp: SchedulerInput,
    requests: Dict[str, List[ModelInstance]],
):
    fastest_edf_schedule = {}
    fastest_edf_schedule[0] = []

    req_by_deadline = [
        (k, v)
        for k, v in sorted(requests.items(), key=lambda x: inp.deadlines[x[0]])
    ]

    for req in req_by_deadline:
        fastest_edf_schedule[0].append(
            ModelAssignment(job_id=req[0], model=get_fastest_model(req[1]))
        )
    return fastest_edf_schedule

def assign_next_instance(
    inp: SchedulerInput,
    candidates: Dict[str, List[ModelInstance]],
    start_time: float,
    last_mod: Optional[ModelInstance],
) -> ModelAssignment:
    #req_by_deadline = [
    #    (k, v)
    #    for k, v in sorted(candidates.items(), key=lambda x: inp.deadlines[x[0]])
    #]
    req_by_deadline = list(reversed([
        (k, v)
        for k, v in sorted(candidates.items(), key=lambda x: inp.get_priority_by_key(x[0]))
    ]))

    #fastest_edf_schedule = {}
    #fastest_edf_schedule[0] = []
    #for req in req_by_deadline:
    #    # TODO: HACK!! (just one worker)
    #    fastest_edf_schedule[0].append(
    #        ModelAssignment(job_id=req[0], model=get_fastest_model(req[1]))
    #    )

    # Let's take the requests in order, but select a model based on the "maximum expected utility"
    # TODO: I think this is more intuitive if you use a priority model, since that will capture which
    # requests will benefit the "most".
    # Or is there a way to capture that here instead of just looking at the effects of the "Fastest"
    # model for each request?

    # This might be OK... let's give it a whirl.

    last_mod = None  # TODO: not sure if we can do anything with this
    next_req_key = req_by_deadline[0][0]
    next_req_mods = req_by_deadline[0][1]
    remaining_req = copy.deepcopy(req_by_deadline)
    del remaining_req[0]

    best_mod = None
    best_avg_util = 0.0

    for m in next_req_mods:
        sum_util = penalized_util(
            m, last_mod, inp.get_model_utility(next_req_key, m),
            start_time, inp.deadlines[next_req_key],
        )
        for req in remaining_req:
            sum_util += inp.get_max_achievable_util(
                req[0], start_time + m.get_latency(last_mod)
            )
        avg_util = sum_util / (1.0 + len(remaining_req))

        if best_mod is None or avg_util > best_avg_util:
            best_mod, best_avg_util = m, avg_util

    return ModelAssignment(job_id=next_req_key, model=best_mod)


def assign_next_instance2(
    inp: SchedulerInput,
    candidates: Dict[str, List[ModelInstance]],
    start_time: float,
    last_mod: Optional[ModelInstance],
) -> ModelAssignment:
    
    def get_exp_latency(candidate) -> float:
        # Assumes uniform distribution
        return statistics.mean(
            [m.get_latency(None) for m in candidate[1]]
        )

    # OK, do the (linear) work upfront for computing a statistic.
    # This could be passed as input to this function, but keep the interface the same for now.
    # fastest_sched = get_fastest_edf_schedule(inp, candidates)
    # gainz_per_sec = []
    # for assigned in fastest_sched[0]:
    #     pass
    # Now go through the candidate models and access their impact (in constant time)
    # mods = candidates[fastest_sched[0][0].job_id]
    # for m in mods:
    #     pass

    #req_by_deadline = [
    #    (k, v)
    #    for k, v in sorted(candidates.items(), key=lambda x: inp.deadlines[x[0]])
    #]
    req_by_deadline = list(reversed([
        (k, v)
        for k, v in sorted(candidates.items(), key=lambda x: inp.get_priority_by_key(x[0]))
    ]))
    last_mod = None  # TODO: not sure if we can do anything with this
    next_req_key = req_by_deadline[0][0]
    next_req_mods = req_by_deadline[0][1]
    remaining_req = copy.deepcopy(req_by_deadline)
    del remaining_req[0]

    best_mod = None
    best_avg_util = 0.0

    for m in next_req_mods:
        sum_util = penalized_util(
            m, last_mod, inp.get_model_utility(next_req_key, m),
            start_time, inp.deadlines[next_req_key],
        )
        #print("start:",exp_start_time)
        exp_start_time = start_time + m.get_latency(last_mod)
        for req in remaining_req:
            #print("exp_start:", exp_start_time)
            sum_util += inp.get_max_achievable_util(req[0], exp_start_time)
            exp_start_time += get_exp_latency(req)
        #print("end:",exp_start_time)
        avg_util = sum_util / (1.0 + len(remaining_req))

        if best_mod is None or avg_util > best_avg_util:
            best_mod, best_avg_util = m, avg_util
    #sys.exit(0)
    return ModelAssignment(job_id=next_req_key, model=best_mod)


def instance_scheduler(inp: SchedulerInput) -> Schedule:
    """TODO: Simulate multiple worker IDs"""
    last_mod = None
    start_time: float = 0
    schedule: Schedule = {}
    schedule[0] = []
    requests = copy.deepcopy(inp.requests)

    while requests:
        assigned = assign_next_instance2(inp, requests, start_time, last_mod)
        schedule[0].append(assigned)

        start_time += assigned.model.get_latency(last_mod)
        del requests[assigned.job_id]
        last_mod = assigned.model
    
    return schedule

InstanceScheduler: Scheduler = instance_scheduler
