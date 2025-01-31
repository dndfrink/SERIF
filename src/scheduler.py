import datetime
import statistics
import time
import sys
import json
from torch.multiprocessing import Queue, Process
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy
from typing import List, Dict, Any, Union, Tuple
from src.get_impl import get_schedule_impl
from src.config_parser import SystemConfig, DataConfig, SystemConfigDecoder, DataConfigDecoder
from src.scheduler_types import Schedule, SchedulerInput, evaluate_schedule
from .request import (
    InferenceRequest,
    TerminationRequest,
)


def pop_all_off_queue(
        inq: Queue) -> Tuple[List[Union[InferenceRequest, TerminationRequest]], bool]:
    # TODO: investigate changing this max jobs size
    MAX_JOBS = 16
    requests = []
    last_batch = False
    while len(requests) < MAX_JOBS:
        try:
            req = inq.get(timeout=0.005)
            if isinstance(req, TerminationRequest):
                last_batch = True
            else:
                requests.append(req)
        except Exception as e:
            break
    return requests, last_batch


def scheduler_entrypoint(system_config: SystemConfig,
                         data_config: DataConfig, input_queue: Queue):
    print("Scheduler entry")
    proxies = {
        node: ServerProxy(node.url, allow_none=True) for node in system_config.get_executors()
    }
    sleep_time = system_config.scheduler_sleep
    batch_sizes = []
    sched_utilities = []
    sched_violations = []
    cum_sched_overhead = 0.0  # track for metrics purposes
    avg_sched_overhead = 0.0
    num_schedules = 0
    use_data_awareness = system_config.data_aware
    verbose = False
    last_schedule = None
    last_batch = False

    while not last_batch:
        batch, last_batch = pop_all_off_queue(input_queue)
        if len(batch) > 0:
            start_time = time.time()
            start_date = datetime.datetime.utcnow()
            req_model_map = {req.token: req.task.get_models() for req in batch}
            req_map = {req.token: req for req in batch}
            delay = {}
            for node, proxy in proxies.items():
                delay[node] = proxy.get_queue_latency()

            # TODO, this should probably use the deadline datetime
            # TODO: more sophisticated understanding for time remaining
            deadlines = {
                req.token: max(
                    (req.get_deadline() - start_date).total_seconds() -
                    avg_sched_overhead, 0.0
                )
                for req in batch
            }
            data_aware_probs = {
                f"{req.token}-probs": req.class_probs for req in batch}
            kv_cache_info = {
                req.token: req._modalities_cached for req in batch}
            scheduler_input = SchedulerInput(
                requests=req_model_map,
                last_schedule=last_schedule,
                executors=system_config.get_executors(),
                use_data_awareness=use_data_awareness,
                data_cache=data_aware_probs,
                deadlines=deadlines,
                executor_delay=delay,
                retrieval_latencies=data_config.get_retrieval_latencies(),
                kv_cache_info=kv_cache_info,
            )
            schedule: Schedule = get_schedule_impl(
                system_config.scheduler_type)(scheduler_input)

            if verbose:
                import pprint

                pprint.pprint(scheduler_input, indent=4, width=1)

            for executor in schedule:
                for assignment in schedule[executor]:
                    req = req_map[assignment.job_id]
                    req.assigned_model = assignment.model
                    req.assigned_server = executor.name
                    proxies[executor].forward_request(req)
                    probs = [] if not data_aware_probs else data_aware_probs[f"{
                        req.token}-probs"]
                    if verbose:
                        print(probs)

            u, v = evaluate_schedule(
                model_schedule=schedule,
                data_cache=data_aware_probs,
                use_data_awareness=use_data_awareness,
                deadlines=deadlines,
                last_schedule=last_schedule,
                executor_delay=delay,
                retrieval_latencies=scheduler_input.retrieval_latencies,
                kv_cache_info=scheduler_input.kv_cache_info,
            )
            sched_utilities.append(u)
            sched_violations.append(v)
            last_schedule = schedule
            batch_sizes.append(len(batch))
            scheduler_time = time.time() - start_time
            cum_sched_overhead += time.time() - start_time
            avg_sched_overhead = ((avg_sched_overhead * num_schedules) + scheduler_time) / (
                num_schedules + 1
            )
            num_schedules += 1
        else:
            time.sleep(sleep_time)

    print("Scheduler results:")
    print("  Data Aware:           ", str(use_data_awareness))
    print("  Scheduling Overhead:  ", str(cum_sched_overhead))
    print("  Average Batch Size:   ", str(statistics.mean(batch_sizes)))
    print("  (Expected) Utility:   ", str(statistics.mean(sched_utilities)))
    print("  (Expected) Violations:", str(statistics.mean(sched_violations)))

    for node, proxy in proxies.items():
        proxy.stop_service()


class SchedulerService:
    def __init__(self, system_config: SystemConfig, data_config: DataConfig):
        self.system_config = system_config
        self.queue = Queue()
        self.data_config = data_config
        self.num_servers_stopped = 0
        self.scheduler = None
        self.average_accuracy = 0.0
        self.average_latency = 0.0
        self.numReqs = 0.0

    def start_scheduler(self):
        self.scheduler = Process(
            target=scheduler_entrypoint,
            args=(self.system_config, self.data_config, self.queue),
            daemon=True,
        )
        self.scheduler.start()

    def enqueue(self, request: Dict[str, Any]):
        self.queue.put(InferenceRequest.from_dict(request))

    def server_stopped(self):
        self.num_servers_stopped += 1
        if self.num_servers_stopped >= len(self.data_config.servers):
            self.queue.put(TerminationRequest())
            self.scheduler.join()


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 -m src.scheduler.py <systemConfigPath> <dataConfigPath>")
        exit(1)
    system_config_file = sys.argv[1]
    data_config_file = sys.argv[2]

    with open(system_config_file, "r") as fh:
        system_config: SystemConfig = json.load(fh, cls=SystemConfigDecoder)
    with open(data_config_file, "r") as fh:
        data_config: DataConfig = json.load(fh, cls=DataConfigDecoder)

    with SimpleXMLRPCServer(
        ("0.0.0.0", int(system_config.scheduler_node.port)), allow_none=True
    ) as server:
        server.register_instance(SchedulerService(system_config, data_config))
        print("Serving at port " + system_config.scheduler_node.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
