import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Set

from redis.cluster import ClusterNode
from src.types import Application, ModelInstance
from src.data_types import DataModality, RequestGenerationNode

from src.system_types import (
    ExecutorNode,
    RedisNode,
    SchedulerNode,
)


class SystemConfig:
    def __init__(
        self,
        seed: int,
        duration_in_secs: int,
        scheduler_type: str,
        scheduler_sleep: float,
        data_aware: bool,
        compute_nodes: List[ExecutorNode],
        redis_nodes: List[RedisNode],
        scheduler_node: SchedulerNode,
    ) -> None:
        self.seed: float = seed if 0 != seed else time.time_ns() % (2**32)
        self.duration_in_secs: int = duration_in_secs
        self.scheduler_type: str = scheduler_type
        self.scheduler_sleep: float = scheduler_sleep
        self.data_aware: bool = data_aware
        self.compute_nodes: List[ExecutorNode] = compute_nodes
        self.redis_nodes: List[RedisNode] = redis_nodes
        self.scheduler_node: SchedulerNode = scheduler_node

    def get_compute_node_by_name(self, name: str):
        for node in self.compute_nodes:
            if node.name == name:
                return node
        return None

    def get_redis_cluster_startup_list(self) -> List[ClusterNode]:
        nodes = []
        for node in self.redis_nodes:
            for port in node.ports:
                nodes.append(ClusterNode(node.ip_address, int(port)))
        return nodes

    def get_executors(self) -> List[ExecutorNode]:
        return self.compute_nodes

    def __repr__(self) -> str:
        return (
            f"SystemConfig(seed={self.seed}, "
            f"duration_in_secs={self.duration_in_secs}, "
            f"scheduler_type={self.scheduler_type}, "
            f"scheduler_sleep={self.scheduler_sleep}, "
            f"data_aware={self.data_aware}, "
            f"compute_nodes={self.compute_nodes}, "
            f"redis_nodes={self.redis_nodes})"
        )


class SystemConfigDecoder(json.JSONDecoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict[str, Any]) -> SystemConfig:
        if "compute-nodes" in obj:
            compute_nodes = [
                ExecutorNode(
                    node.get("name"),
                    node.get("ip-address"),
                    node.get("server-port"),
                )
                for node in obj.get("compute-nodes", [])
            ]
            redis_nodes = [
                RedisNode(
                    node.get("ip-address"),
                    node.get("ports"),
                )
                for node in obj.get("redis-nodes", [])
            ]
            scheduler_node = SchedulerNode(
                obj.get("scheduler-node").get("ip-address"),
                obj.get("scheduler-node").get("port"),
            )
            system_config = SystemConfig(
                seed=obj.get("seed"),
                duration_in_secs=obj.get("duration-in-secs"),
                scheduler_type=obj.get("scheduler-type"),
                scheduler_sleep=obj.get("scheduler-sleep"),
                data_aware=obj.get("data-aware"),
                compute_nodes=compute_nodes,
                redis_nodes=redis_nodes,
                scheduler_node=scheduler_node,
            )
            return system_config
        else:
            return obj


class ApplicationConfig:
    def __init__(self, applications: List[Application]):
        self.applications = applications

    def __repr__(self):
        return f"ApplicationConfig(applications={self.applications})"


class ApplicationConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict[str, Any]) -> Any:
        if "applications" in obj:
            applications = [
                Application(
                    name=app.get("name"),
                    sla_latency=app.get("sla-latency"),
                    sample_duration=app.get("sample-duration"),
                    utility_matrix=app.get("utility-matrix"),
                    prior=app.get("prior"),
                    model_profiles=[
                        ModelInstance(
                            name=profile.get("name"),
                            task=app.get("name"),
                            use_profile=profile.get("use-profile"),
                            modality=profile.get("modality"),
                            utility_matrix=app.get("utility-matrix"),
                            confusion_matrix=profile.get("cmat"),
                            latencies=profile.get("latencies"),
                            impl={
                                "weights": profile.get("model-data"),
                                "redis-key": profile.get("redis-key"),
                            },
                        )
                        for profile in app.get("model-profiles", [])
                    ],
                )
                for app in obj.get("applications", [])
            ]
            return ApplicationConfig(applications)
        else:
            return obj


@dataclass
class DataConfig:
    modalities: List[DataModality]
    servers: List[RequestGenerationNode]
    num_groups: int

    def __init__(
        self,
        modalities: List[DataModality],
        servers: List[RequestGenerationNode],
        num_groups: int,
    ):
        self.modalities = modalities
        self.servers = servers
        self.num_groups = num_groups
        self.validate_stream_mappings()

    def assign_ranges_for_servers(self) -> Dict[str, List[str]]:
        # Check if we have any non-synthetic modalities
        non_synthetic_modality = next(
            (m for m in self.modalities if not m.use_synthetic_data),
            None,
        )
        assignments = None
        if non_synthetic_modality is None:
            # All modalities are synthetic - divide based on worker counts
            workers_per_server = {
                server.name: server.num_workers for server in self.servers
            }
            num_total_workers = sum(workers_per_server.values())
            assignments: Dict[str, Set[str]] = {}
            current_start = 0
            for server, workers in workers_per_server.items():
                ratio = workers / num_total_workers
                id_count = round(self.num_groups * ratio)
                if server == list(workers_per_server.keys())[-1]:
                    end = self.num_groups
                else:
                    end = current_start + id_count

                assignments[server] = {str(i)
                                       for i in range(current_start, end)}
                current_start = end
        else:
            # We have non-synthetic data - use the pre-validated grouping
            # We only need to look at one non-synthetic modality since
            # validation ensures consistency
            assignments = {
                server.name: set(
                    server.streams.get(
                        non_synthetic_modality.modality, {}
                    ).keys()
                )
                for server in self.servers
            }

            # Remove any empty assignments
            assignments = {k: v for k, v in assignments.items() if v}

        return {server: list(groups) for server, groups in assignments.items()}

    def validate_stream_mappings(self) -> None:
        expected_groups = set(str(i) for i in range(self.num_groups))

        # Get list of non-synthetic modalities
        non_synthetic_modalities = [
            m for m in self.modalities if not m.use_synthetic_data
        ]
        if not non_synthetic_modalities:
            return  # No validation needed if all modalities are synthetic

        # First validate complete coverage for each modality
        for modality in non_synthetic_modalities:
            available_groups: Set[str] = set()
            for server in self.servers:
                if modality.modality in server.streams:
                    available_groups.update(
                        server.streams[modality.modality].keys())
            if available_groups != expected_groups:
                raise ValueError(
                    f"Groups mismatch for modality {modality.modality}. "
                    f"Expected groups 0-{self.num_groups - 1}, got: "
                    f"{sorted(int(g) for g in available_groups)}"
                )

        # Then validate that each server has consistent group IDs across its
        # non-synthetic modalities
        for server in self.servers:
            # Get the set of group IDs for each non-synthetic modality
            modality_groups: Dict[str, Set[str]] = {}

            for modality in non_synthetic_modalities:
                if modality.modality in server.streams:
                    modality_groups[modality.modality] = set(
                        server.streams[modality.modality].keys()
                    )
                else:
                    modality_groups[modality.modality] = set()

            # If server has any non-synthetic modalities, all must have same
            # groups
            if modality_groups:
                first_modality = non_synthetic_modalities[0].modality
                first_groups = modality_groups[first_modality]

                for modality in non_synthetic_modalities[1:]:
                    if modality_groups[modality.modality] != first_groups:
                        raise ValueError(
                            f"Inconsistent group IDs for server {
                                server.name}. "
                            f"Modality {first_modality} has groups "
                            f"{sorted(int(g) for g in first_groups)}, "
                            f"but modality {modality.modality} has groups "
                            f"{sorted(int(g)
                                      for g in modality_groups[modality.modality])}"
                        )

    def get_modality_by_name(self, modality_name: str) -> DataModality:
        """Helper function to find a modality object by name."""
        return next(
            (m for m in self.modalities if m.modality == modality_name),
            None,
        )

    def get_retrieval_latencies(self):
        latencies = {}
        for modality in self.modalities:
            latencies[modality.modality] = modality.retrieval_latencies
        return latencies

    def get_data_server_by_name(self, name: str) -> RequestGenerationNode:
        for server in self.servers:
            if server.name == name:
                return server
        return None

    def get_data_server_urls(self) -> List[str]:
        urls = []
        for server in self.servers:
            urls.append(f"http://{server.ip_address}:{server.port}")
        return urls

    def get_cache_data_for_server(
        self, server: Union[RequestGenerationNode, ExecutorNode]
    ) -> Dict[str, Dict[str, int]]:
        modality_map = {}
        if isinstance(server, RequestGenerationNode):
            for modality in self.modalities:
                modality_map[modality.modality] = {
                    "size": modality.cache_entry_size
                }
                modality_map[modality.modality]["entries"] = (
                    server.get_num_entries_for_modality(modality.modality)
                )
            return modality_map
        elif isinstance(server, ExecutorNode):
            for data_server in self.servers:
                if data_server.ip_address == server.ip_address:
                    for modality in self.modalities:
                        modality_map[modality.modality] = {
                            "size": modality.cache_entry_size
                        }
                        modality_map[modality.modality]["entries"] = (
                            data_server.get_num_entries_for_modality(
                                modality.modality
                            )
                        )
                    return modality_map
        return None


class DataConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict[str, Any]) -> DataConfig:
        if "data-modalities" in obj:
            modalities = [
                DataModality(
                    modality=modality.get("name"),
                    shape=modality.get("synthetic-shape"),
                    use_synthetic_data=modality.get("use-synthetic-data"),
                    cache_entry_size=modality.get("cache-entry-size"),
                    retrieval_latencies=modality.get("retrieval-latencies"),
                )
                for modality in obj.get("data-modalities", [])
            ]
            servers = [
                RequestGenerationNode(
                    name=server.get("name"),
                    ip_address=server.get("ip-address"),
                    port=server.get("port"),
                    num_workers=server.get("workers"),
                    streams=server.get("streams"),
                    cache_entries_per_modality=server.get(
                        "cache-entries-per-modality"
                    ),
                )
                for server in obj.get("servers", [])
            ]
            num_groups = obj.get("num-groups")
            return DataConfig(
                modalities=modalities,
                servers=servers,
                num_groups=num_groups,
            )
        else:
            return obj
