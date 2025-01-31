import concurrent.futures
import itertools
import random

from functools import partial
from typing import Tuple

from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment, evaluate_schedule

child_req_ids = None
child_inp : SchedulerInput = None

def initwrapper(initfunc, initargs, f, x):
    global child_req_ids, child_inp
    if not child_inp:
        initfunc(*initargs)
    return f(x)

def init_worker(req_ids, inp):
    global child_req_ids, child_inp
    child_req_ids = req_ids
    child_inp = inp

def initmap(executor, initializer, initargs, f, it):
    return executor.map(partial(initwrapper, initializer, initargs, f), it)

def batch_eval_schedules(batch_perm) -> Tuple[Schedule, float]:
    global child_req_ids, child_inp

    best_schedule = None
    best_utility = -1000000
    batch_perm_list = list(batch_perm)
    mod_seq = [range(len(x[1])) for x in batch_perm_list]

    for mod_indexes in itertools.product(*mod_seq):
        tmp_sched = {}
        tmp_sched[0] = []

        for i in range(len(mod_indexes)):
            tmp_sched[0].append(
                ModelAssignment(
                    job_id=child_req_ids[batch_perm_list[i][0]],
                    model=batch_perm_list[i][1][mod_indexes[i]],
                )
            )

        exp_util, violation = evaluate_schedule(
            tmp_sched, child_inp.data_cache, child_inp.use_data_awareness,
            child_inp.deadlines, child_inp.last_schedule, child_inp.executor_delay,
        )
        prefer_this_schedule = True if exp_util > best_utility else False
        if best_schedule is None or prefer_this_schedule:
            best_schedule = tmp_sched
            best_utility = exp_util

    return best_schedule, best_utility

def brute_force_scheduler(inp: SchedulerInput) -> Schedule:
    batch_tasks = []
    req_ids = list(inp.requests.keys())

    for i, req in enumerate(req_ids):
        batch_tasks.append([i, inp.requests[req]])

    best_schedule = None
    best_utility = -1000000

    batch_perms = itertools.permutations(batch_tasks)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        output = initmap(pool, init_worker, (req_ids, inp), batch_eval_schedules, batch_perms)
        for res in output:
            prefer_this_schedule = True if res[1] > best_utility else False
            if best_schedule is None or prefer_this_schedule:
                best_schedule = res[0]
                best_utility = res[1]

    return best_schedule


BruteForceParallelScheduler: Scheduler = brute_force_scheduler
