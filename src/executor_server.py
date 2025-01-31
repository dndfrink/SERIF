"""Executor service implementation for handling inference requests."""

from xmlrpc.server import SimpleXMLRPCServer
from typing import List
import sys
import json
import torch.multiprocessing as mp

from src.config_parser import (
    ApplicationConfig,
    SystemConfig,
    ApplicationConfigDecoder,
    SystemConfigDecoder,
    DataConfig,
    DataConfigDecoder,
)
from src.executor import executor_entrypoint
from src.request_queue import RequestQueue
from src.request import TerminationRequest, InferenceRequest


class ExecutorService:
    def __init__(
        self,
        system_config: SystemConfig,
        application_config: ApplicationConfig,
        data_config: DataConfig,
        name: str,
    ):
        self._system_config = system_config
        self._application_config = application_config
        self._data_config = data_config
        self._node_info = self._system_config.get_compute_node_by_name(name)
        self._worker = None
        self._queue = RequestQueue(data_config.get_retrieval_latencies())
        self._console_lock = mp.Lock()
        self._stats = mp.Array('d', 3)

    def start_service(self):
        for i in range(3):
            self._stats[i] = 0.0

        self._worker = mp.Process(
            target=executor_entrypoint,
            args=(
                self._system_config,
                self._application_config,
                self._data_config,
                self._queue,
                self._node_info,
                self._console_lock,
                self._system_config.seed,
                self._stats,
            ),
            daemon=True,
        )

        self._worker.start()

    def stop_service(self):
        self._queue.put(TerminationRequest())
        self._worker.join()

    def forward_request(self, request: InferenceRequest):
        if isinstance(request, dict):
            request = InferenceRequest.from_dict(request)
        self._queue.put(request)

    def get_queue_latency(self) -> float:
        return self._queue.get_queue_latency()

    def get_stats(self) -> List:
        return list(self._stats[:])


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 executor_server.py <server_name> "
            "<system_config_file_path> <application_config_file_path> "
            "<data_config_file_path>"
        )
        sys.exit()

    name = sys.argv[1]
    sys_config_file = sys.argv[2]
    app_config_file = sys.argv[3]
    data_config_file = sys.argv[4]

    with open(sys_config_file, "r") as fh:
        system_config: SystemConfig = json.load(fh, cls=SystemConfigDecoder)

    with open(app_config_file, "r") as fh:
        application_config: ApplicationConfig = json.load(
            fh, cls=ApplicationConfigDecoder
        )

    with open(data_config_file, "r") as fh:
        data_config: DataConfig = json.load(fh, cls=DataConfigDecoder)

    port = system_config.get_compute_node_by_name(name).port

    with SimpleXMLRPCServer(("0.0.0.0", int(port)), allow_none=True) as server:
        server.register_instance(
            ExecutorService(
                system_config,
                application_config,
                data_config,
                name)
        )
        print(f"Serving at port {port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
