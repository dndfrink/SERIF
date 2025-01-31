import copy
import random

from typing import List, Tuple

from sneakpeek.scheduling import get_fastest_model
from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment, evaluate_schedule

from .locally_optimal import locally_optimal_instance


def reorder_requests2(inp: SchedulerInput):
    req_by_deadline = inp.get_sorted_requests_by_deadline()
    req_map = {req[0]: req for req in req_by_deadline}

    fastest_edf_schedule = {}
    fastest_edf_schedule[0] = []

    for req in req_by_deadline:
        mod_list = [ get_fastest_model(req[1]) ]
        token = req[0]
        worker_id, mod = locally_optimal_instance(
            inp, fastest_edf_schedule, token, mod_list, inp.deadlines[token]
        )
        fastest_edf_schedule[worker_id].append(
            ModelAssignment(job_id=token, model=mod)
        )

    # OK, need to figure out how to generate new schedules with two
    # elements swapped. Then compare their utility / violations.
    # Just do one executor for now.
    n_req = len(req_by_deadline)
    best_sched = fastest_edf_schedule
    best_util, _ = evaluate_schedule(
        best_sched, inp.data_cache, inp.use_data_awareness, inp.deadlines, None, inp.executor_delay,
    )

    for _ in range(n_req):
        for i in reversed(range(1, n_req)):
            behind_entry = best_sched[0][i]
            ahead_entry = best_sched[0][i-1]
            behind_priority = inp.get_priority_by_key(behind_entry.job_id)
            ahead_priority = inp.get_priority_by_key(ahead_entry.job_id)
            if behind_priority > ahead_priority:
                # See if a switch is possible
                next_sched = copy.deepcopy(best_sched)
                next_sched[0][i-1], next_sched[0][i] = behind_entry, ahead_entry
                next_util, _ = evaluate_schedule(
                    next_sched, inp.data_cache, inp.use_data_awareness, inp.deadlines, None, inp.executor_delay,
                )
                if next_util >= (best_util - 0.00001):
                    best_sched = next_sched
                    best_util = next_util

    return [req_map[x.job_id] for x in best_sched[0]]


def reorder_requests(inp: SchedulerInput):
    req_by_deadline = inp.get_sorted_requests_by_deadline()

    # OK let's just assume a single worker for now. Eventually, we should
    # generate the "fastest" schedule and use those worker assignments.
    agg = 0.0
    timed_list = []
    last_mod = None
    for req in req_by_deadline:
        timed_list.append((agg, req))
        mod = get_fastest_model(req[1])
        agg += mod.get_inference_latency()
        agg += mod.get_context_switch_latency() if last_mod is None or last_mod != mod else 0.0

    # Time for bubble sort, I guess.
    for _ in range(len(timed_list)):
        for i in reversed(range(1, len(timed_list))):
            behind_entry = timed_list[i]
            ahead_entry = timed_list[i-1]
            behind_priority = inp.get_priority_by_key(behind_entry[1][0])
            ahead_priority = inp.get_priority_by_key(ahead_entry[1][0])
            if behind_priority > ahead_priority:
                # See if a switch is possible
                later_start = behind_entry[0]
                deadline = inp.deadlines[ahead_entry[1][0]]
                if later_start + get_fastest_model(ahead_entry[1][1]).get_inference_latency() < deadline:
                    timed_list[i-1], timed_list[i] = behind_entry, ahead_entry

    return [x[1] for x in timed_list]


def heuristic_scheduler(inp: SchedulerInput) -> Schedule:
    schedule = { i:[] for i in range(inp.num_executors)}
    #requests = reorder_requests(inp)
    requests = reorder_requests2(inp)

    print("Heuristic PRIORITIES:")
    for req in requests:
        token = req[0]
        mod_list = req[1]
        low, high = inp.get_priority_range_by_key(token)
        print("   ","low:",low,"high:",high,"range:",high-low)
        worker_id, mod = locally_optimal_instance(
            inp, schedule, token, mod_list, inp.deadlines[token]
        )
        schedule[worker_id].append(
            ModelAssignment(job_id=token, model=mod)
        )

    return schedule

HeuristicScheduler: Scheduler = heuristic_scheduler
