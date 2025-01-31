from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy
from typing import Dict, Tuple
import sys
import json
import torch.multiprocessing as mp

from src.config_parser import (
    ApplicationConfig,
    SystemConfig,
    DataConfig,
    ApplicationConfigDecoder,
    SystemConfigDecoder,
    DataConfigDecoder,
)
from src.data_io_task import io_task_entrypoint, kv_garbage_collector_entrypoint
from src.kvcache import KVCache


class DataService:
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
        self._server_info = data_config.get_data_server_by_name(name)
        self._workers = []
        self._garbage_collector = None
        self._console_lock = mp.Lock()
        self._local_kv_store = KVCache(
            self._data_config.get_cache_data_for_server(self._server_info)
        )

    def start_service(self, range_map: Dict[str, Tuple]):
        self._server_info.group_range = range_map[self._server_info.name]
        data_divisions = self._server_info.get_worker_divisions()

        self._workers = [
            mp.Process(
                target=io_task_entrypoint,
                args=(
                    self._system_config,
                    self._application_config,
                    self._data_config,
                    self._server_info,
                    self._console_lock,
                    data_divisions[i],
                    self._system_config.seed + i,
                ),
                daemon=True,
            )
            for i in range(self._server_info.num_workers)
        ]

        self._garbage_collector = mp.Process(
            target=kv_garbage_collector_entrypoint,
            args=(self._data_config, self._server_info),
            daemon=True,
        )

        for worker in self._workers:
            worker.start()

        self._garbage_collector.start()

    def join(self):
        for worker in self._workers:
            worker.join()

        # Send a kill signal since this process runs forever for simplicity
        self._garbage_collector.terminate()

        # Clean up the shared memory
        self._local_kv_store.kv_store.close()
        self._local_kv_store.kv_store.unlink()
        self._local_kv_store.metadata.close()
        self._local_kv_store.metadata.unlink()

        # Notify the scheduler that a data server has stopped sending data
        scheduler_proxy = ServerProxy(
            self._system_config.scheduler_node.url, allow_none=True
        )
        scheduler_proxy.server_stopped()


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 data_server.py <server_name> "
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

    port = data_config.get_data_server_by_name(name).port

    with SimpleXMLRPCServer(("0.0.0.0", int(port)), allow_none=True) as server:
        server.register_instance(
            DataService(system_config, application_config, data_config, name)
        )
        print(f"Serving at port {port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
