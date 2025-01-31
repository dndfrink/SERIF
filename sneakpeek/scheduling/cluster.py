import copy
import itertools
import statistics

from dataclasses import dataclass
from typing import List, Tuple, Dict

from src.types import ModelInstance
from src.scheduler_types import Schedule, Scheduler, SchedulerInput, ModelAssignment, evaluate_schedule

@dataclass
class ClusterInfo:
    task: str
    models: List[ModelInstance]
    avg_deadline: float = 0.0

    def __init__(self, req : Tuple[str,List[ModelInstance]]):
        self.task = req[1][0].task
        self.models = req[1]

# A bit of a hack; this should really be given as input
def get_task_for_req(req : Tuple[str,List[ModelInstance]]) -> str:
    return req[1][0].task

def get_clusters_and_models(inp: SchedulerInput, sorted_req):
    clusters : Dict[str, List] = {}
    cluster_info : Dict[str, ClusterInfo] = {}
    models_by_task = {}

    if not inp.use_data_awareness:
        for req in sorted_req:
            task = get_task_for_req(req)
            if task not in clusters:
                clusters[task] = []
                models_by_task[task] = req[1]
                cluster_info[task] = ClusterInfo(req)
            clusters[task].append(req)
    else:
        for req in sorted_req:
            task = get_task_for_req(req)
            data_key = f"{req[0]}-probs"
            if data_key in inp.data_cache:
                probs = inp.data_cache[data_key]
                task = f"{task}-{str(probs.index(max(probs)))}"

            if task not in clusters:
                clusters[task] = []
                models_by_task[task] = req[1]
                cluster_info[task] = ClusterInfo(req)
            clusters[task].append(req)

    for c_key in clusters:
        deadlines = []
        for req in clusters[c_key]:
            deadlines.append(inp.deadlines[req[0]])
        cluster_info[c_key].avg_deadline = statistics.mean(deadlines)

    return clusters, models_by_task, cluster_info


def locally_optimal_instance(inp : SchedulerInput, current_schedule : Schedule, req_list, model_list):
    candidates = [
        (e_idx, mod)
        for e_idx in range(inp.num_executors)
        for mod in model_list
    ]

    best_candidate = None
    best_utility = -1000000

    for candidate in candidates:
        tmp_sched = copy.deepcopy(current_schedule)
        for req in req_list:
            tmp_sched[candidate[0]].append(
                ModelAssignment(job_id=req[0], model=candidate[1])
            )
        exp_util, violation = evaluate_schedule(
            tmp_sched, inp.data_cache, inp.use_data_awareness,
            inp.deadlines, inp.last_schedule, inp.executor_delay,
        )

        # I think we determined all candidates are feasible above...
        prefer_this_schedule = True if exp_util > best_utility else False

        if best_candidate is None or prefer_this_schedule:
            best_candidate = candidate
            best_utility = exp_util

    return best_candidate


def generic_cluster_locally_optimal(inp: SchedulerInput, sorted_req) -> Schedule:
    schedule = { i:[] for i in range(inp.num_executors)}
    clusters, models_by_task, cluster_info = get_clusters_and_models(inp, sorted_req)

    ordered_keys = [
        k
        for k in sorted(clusters.keys(), key=lambda x: cluster_info[x].avg_deadline)
    ]

    for c_key in ordered_keys:
        executor_idx, mod = locally_optimal_instance(
            inp, schedule, clusters[c_key], models_by_task[c_key],
        )
        for req in clusters[c_key]:
            schedule[executor_idx].append(ModelAssignment(job_id=req[0], model=mod))

    return schedule


def generic_cluster_scheduler(inp: SchedulerInput, sorted_req) -> Schedule:
    schedule = { node :[] for node in inp.executors}
    clusters, models_by_task, cluster_info = get_clusters_and_models(inp, sorted_req)

    best_schedule = None
    best_utility = -1000000

    c_keys = clusters.keys()
    for cluster_order in itertools.permutations(c_keys):
        ordered_tasks = list(cluster_order)
        mod_seq = [range(len(models_by_task[t])) for t in ordered_tasks]

        for executor in inp.executors:
            for mod_indexes in itertools.product(*mod_seq):
                tmp_sched = {}
                tmp_sched[executor] = []

                for i in range(len(mod_indexes)):
                    task_id = ordered_tasks[i]
                    task_mod = models_by_task[task_id][mod_indexes[i]]

                    for req in clusters[task_id]:
                        tmp_sched[executor].append(
                            ModelAssignment(job_id=req[0],model=task_mod)
                        )

                exp_util, violation = evaluate_schedule(
                    tmp_sched, inp.data_cache, inp.use_data_awareness,
                    inp.deadlines, inp.last_schedule, inp.executor_delay,
                    inp.retrieval_latencies, inp.kv_cache_info
                )
                prefer_this_schedule = True if exp_util > best_utility else False
                if best_schedule is None or prefer_this_schedule:
                    best_schedule = tmp_sched
                    best_utility = exp_util

    return best_schedule

def dynamic_cluster_scheduler(inp: SchedulerInput) -> Schedule:
    requests = inp.get_sorted_requests_by_deadline()
    #clusters, models_by_task, cluster_info = get_clusters_and_models(inp, requests)
    #if len(clusters) > 4:
    #    return generic_cluster_locally_optimal(inp, requests)
    # JSW TODO HACK
    return generic_cluster_scheduler(inp, requests)

def dynamic_priority_cluster_scheduler(inp: SchedulerInput) -> Schedule:
    requests = inp.get_sorted_requests_by_priority()
    #clusters, models_by_task, cluster_info = get_clusters_and_models(inp, requests)
    #if len(clusters) > 4:
    #    return generic_cluster_locally_optimal(inp, requests)
    # JSW TODO HACK
    return generic_cluster_scheduler(inp, requests)


ClusterEDFDynamicScheduler: Scheduler = dynamic_cluster_scheduler
ClusterPriorityDynamicScheduler: Scheduler = dynamic_priority_cluster_scheduler

