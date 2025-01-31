import statistics
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
from src.system_types import ExecutorNode
from src.types import ModelInstance


@dataclass
class ModelAssignment:
    job_id: str
    model: ModelInstance


WorkerId = int
# Keys are workers, values are an ordered list
Schedule = Dict[ExecutorNode, List[ModelAssignment]]

class PriorityType(Enum):
    UTIL_RANGE = 1
    UTIL_MAX = 2
    UTIL_NORMALIZED = 3     # For which model though? Average somehow?
    UTIL_DEADLINE_ADJUSTED = 4
    ACC_DEADLINE_ADJUSTED = 5
    ACC_DEADLINE_AND_LATENCY_ADJUSTED = 6


@dataclass(frozen=True)
class SchedulerInput:
    requests: Dict[str, List[ModelInstance]]
    deadlines: Optional[Dict[str, float]]
    last_schedule: Optional[Schedule]
    executors: List[ExecutorNode]
    data_cache: Dict[str, Any]
    use_data_awareness: bool
    executor_delay: Dict[ExecutorNode, float]
    retrieval_latencies: Dict[str, Dict[str, float]]
    kv_cache_info: Dict[str, Dict[str, str]]
    priority_type: PriorityType = PriorityType.UTIL_DEADLINE_ADJUSTED

    @property
    def num_executors(self):
        return len(self.executors)

    def get_sorted_requests_by_deadline(
            self) -> List[Tuple[str, List[ModelInstance]]]:
        return [
            (k, v)
            for k, v in sorted(self.requests.items(), key=lambda x: self.deadlines[x[0]])
        ]

    def get_max_achievable_util(self, req_key: str,
                                start_time: float) -> float:
        mods = self.requests[req_key]
        max_util = 0.0
        for mod in mods:
            prev_mod = None  # Does this make sense?
            mod_util = penalized_util(
                mod, prev_mod, self.get_model_utility(req_key, mod),
                start_time, self.deadlines[req_key],
            )
            max_util = mod_util if mod_util > max_util else max_util
        return max_util

    def get_avg_expected_util(self, req_key: str, start_time: float) -> float:
        mods = self.requests[req_key]
        utils = []
        for mod in mods:
            prev_mod = None  # Does this make sense?
            mod_util = penalized_util(
                mod, prev_mod, self.get_model_utility(req_key, mod),
                start_time, self.deadlines[req_key],
            )
            utils.append(mod_util)
        return statistics.mean(utils)

    def get_avg_model_statistics(self, req_key: str):
        mods = self.requests[req_key]
        acc = [m.get_inference_accuracy() for m in mods]
        lat = [m.get_latency(None) for m in mods]
        return (
            statistics.mean(acc), statistics.variance(acc),
            statistics.mean(lat), statistics.variance(lat),
            min(lat)
        )

    def get_priority_by_key(self, req_key: str) -> float:
        if self.priority_type == PriorityType.ACC_DEADLINE_ADJUSTED:
            ma, va, ml, vl, minLat = self.get_avg_model_statistics(req_key)
            return (1.0 + va) * math.exp(-self.deadlines[req_key])

        if self.priority_type == PriorityType.ACC_DEADLINE_AND_LATENCY_ADJUSTED:
            ma, va, ml, vl, minLat = self.get_avg_model_statistics(req_key)
            return (1.0 + va) * math.exp(-self.deadlines[req_key] + minLat)

        if self.priority_type == PriorityType.UTIL_DEADLINE_ADJUSTED:
            start_time = 0.0  # TODO: does it make sense to enhance this?
            max_util = self.get_avg_expected_util(req_key, start_time)
            try:
                return max_util / self.deadlines[req_key]
            except ZeroDivisionError:
                print("PAST DEADLINE IN SCHEDULER")
                return max_util * 100

        if self.priority_type == PriorityType.UTIL_RANGE:
            p_range = self.get_priority_range_by_key(req_key)
            return p_range[1] - p_range[0]

        if self.priority_type == PriorityType.UTIL_MAX:
            p_range = self.get_priority_range_by_key(req_key)
            return p_range[1]

        if self.priority_type == PriorityType.UTIL_NORMALIZED:
            norm_utils = []
            utils = []
            lats = []
            for m in self.requests[req_key]:
                mod_util = self.get_model_utility(req_key, m)
                norm_utils.append(mod_util / m.get_inference_latency())
                utils.append(mod_util)
                lats.append(m.get_inference_latency())
            return sum(utils) / sum(lats)

        raise RuntimeError("unsupported priority type")

    def get_model_utility(self, req_key: str, mod: ModelInstance) -> float:
        mk = req_key + "-probs"
        if self.use_data_awareness and mk in self.data_cache:
            return mod.expected_conditional_utility(self.data_cache[mk])
        return mod.expected_utility()

    def get_priority_range_by_key(self, req_key: str) -> Tuple[float, float]:
        fastest_model_utility = -100000.0
        fastest_model_latency = 1000000.0
        highest_utility = -100000.0
        mods = self.requests[req_key]
        for mod in mods:
            mod_util = self.get_model_utility(req_key, mod)
            if mod.get_inference_latency() < fastest_model_latency:
                fastest_model_utility = mod_util
                fastest_model_latency = mod.get_inference_latency()
            highest_utility = max(highest_utility, mod_util)
        return fastest_model_utility, highest_utility

    def get_sorted_requests_by_priority(
            self) -> List[Tuple[str, List[ModelInstance]]]:
        """
        Let's try using the range: fastest model to highest utility.
        We don't really care about the other options.
        """
        return list(reversed([
            (k, v)
            for k, v in sorted(
                self.requests.items(), key=lambda x: self.get_priority_by_key(x[0])
            )
        ]))


def penalized_util(mod: ModelInstance, prev_mod, max_util,
                   start_time, deadline, network_latency=0.0) -> float:
    end_time = start_time + mod.get_latency(prev_mod) + network_latency
    if end_time < (deadline + 0.0001):
        # Calculate how close we are to deadline as a ratio between 0 and 1
        time_until_deadline = deadline - end_time
        deadline_proximity = 1.0 - (time_until_deadline / deadline)
        # Scale network latency penalty based on deadline proximity
        # Maximum penalty is 0.2 when very close to deadline
        network_penalty = 0.2 * deadline_proximity * \
            (network_latency / (network_latency + 1.0))
        return max_util - network_penalty

    # We are going to violate an slo. Reduce the utility accordingly.
    if end_time > (2.0 * deadline):  # lambda = 2.0
        power = 2 if network_latency >= 1 else 0.5
        return -math.pow(network_latency, power)

    def sigmoid(x: float) -> float:
        if x > 0.99999:
            return 1.0
        if x < 0.000001:
            return 0.0
        term = math.pow((x / (1.0 - x)), -3.0)
        return 1.0 / (1 + term)

    sig_inp = 1.0 - ((2.0 * deadline - end_time) / deadline)
    power = 2 if network_latency >= 1 else 0.5
    return (max_util * (1.0 - sigmoid(sig_inp))) - \
        math.pow(network_latency, power)

def evaluate_schedule(
    model_schedule: Schedule,
    data_cache: Dict[str, Any],
    use_data_awareness: bool,
    deadlines: Dict[str, float],
    last_schedule: Optional[Schedule],
    executor_delay: Optional[Dict[ExecutorNode, float]],
    retrieval_latencies : Dict[str, Dict[str, float]],
    kv_cache_info : Dict[str, Dict[str, str]]
) -> Tuple[float, float]:
    """
    Evaluates a given schedule in terms of expected utility.
    returns (expected utility, expected SLO violation)
    """
    # TODO: Current limitations:
    #   - Assumes homogeneous executors
    utils = []
    slo_violation = 0.0
    for executor in model_schedule:
        start_time = (
            executor_delay[executor]
            if executor_delay and executor in executor_delay
            else 0.0
        )
        last_mod = (
            last_schedule[executor][-1].model
            if last_schedule and executor in last_schedule and len(last_schedule[executor]) > 0
            else None
        )
        for assignment in model_schedule[executor]:
            da_key = assignment.job_id + '-probs'
            mod = assignment.model
            deadline = deadlines[assignment.job_id]

            network_latency = 0.0
            try:
                is_req_cached = (executor.ip_address == kv_cache_info[assignment.job_id][mod.modality])
                network_latency = retrieval_latencies[mod.modality][executor.name] if not is_req_cached else 0.0
            except KeyError:
                pass

            if use_data_awareness and da_key in data_cache:
                mod_util = mod.expected_conditional_utility(data_cache[da_key])
            else:
                mod_util = mod.expected_utility()
            mod_util = penalized_util(mod, last_mod, mod_util, start_time, deadline, network_latency)
            utils.append(mod_util)

            latency = mod.get_inference_latency()
            latency += (
                 mod.get_context_switch_latency()
                 if last_mod is None or last_mod != mod
                 else 0.0
            )
            latency += network_latency

            if (start_time + latency) > deadline:
                slo_violation += (start_time + latency) - deadline

            last_mod = mod
            start_time += latency

    util = (
        statistics.mean(utils) if len(utils) > 0
        else 0.0
    )
    return util, slo_violation

Scheduler = Callable[[SchedulerInput], Schedule]