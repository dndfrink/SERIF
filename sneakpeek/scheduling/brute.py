import itertools
import random

from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment, evaluate_schedule 


def brute_force_scheduler(inp: SchedulerInput) -> Schedule:
    batch_tasks = []
    req_ids = list(inp.requests.keys())

    for i, req in enumerate(req_ids):
        batch_tasks.append([i, inp.requests[req]])

    best_schedule = None
    best_utility = -1000000

    for batch_perm in itertools.permutations(batch_tasks):
        batch_perm_list = list(batch_perm)
        mod_seq = [range(len(x[1])) for x in batch_perm_list]

        for mod_indexes in itertools.product(*mod_seq):
            tmp_sched = {}
            tmp_sched[0] = []

            for i in range(len(mod_indexes)):
                tmp_sched[0].append(
                    ModelAssignment(
                        job_id=req_ids[batch_perm_list[i][0]],
                        model=batch_perm_list[i][1][mod_indexes[i]],
                    )
                )

            exp_util, violation = evaluate_schedule(
                tmp_sched, inp.data_cache, inp.use_data_awareness,
                inp.deadlines, inp.last_schedule, inp.executor_delay,
            )
            prefer_this_schedule = True if exp_util > best_utility else False
            if best_schedule is None or prefer_this_schedule:
                best_schedule = tmp_sched
                best_utility = exp_util

    print("\n\n")
    return best_schedule


BruteForceScheduler: Scheduler = brute_force_scheduler
