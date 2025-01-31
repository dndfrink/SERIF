import torch
import torch.multiprocessing as mp

import numpy as np
import sys
import time
import json

from .config_parser import SystemConfigDecoder, ApplicationConfigDecoder, SystemConfig, ApplicationConfig, DataConfig, DataConfigDecoder
from redis import RedisCluster
from xmlrpc.client import ServerProxy


def run_pipeline(system_config: SystemConfig,
                 application_config: ApplicationConfig, data_config: DataConfig) -> None:

    torch.multiprocessing.set_start_method('spawn')

    shared_kv_store = RedisCluster(
        startup_nodes=system_config.get_redis_cluster_startup_list()
    )

    for app in application_config.applications:
        for model in app.model_profiles:
            if None is not model.impl["weights"]:
                with open(model.impl["weights"], "rb") as fh:
                    shared_kv_store.set(model.impl["redis-key"], fh.read())

    scheduler_proxy = ServerProxy(
        system_config.scheduler_node.url,
        allow_none=True)
    scheduler_proxy.start_scheduler()

    executor_proxies = [
        ServerProxy(
            node.url,
            allow_none=True) for node in system_config.get_executors()]
    for proxy in executor_proxies:
        proxy.start_service()

    time.sleep(15)  # Executor startup time

    # TODO: only send mapping for the server being started
    data_server_urls = data_config.get_data_server_urls()
    data_proxies = [ServerProxy(url, allow_none=True)
                    for url in data_server_urls]
    server_to_group_range_map = data_config.assign_ranges_for_servers()

    for proxy in data_proxies:
        proxy.start_service(server_to_group_range_map)

    for proxy in data_proxies:
        proxy.join()

    average_latency = 0.0
    average_accuracy = 0.0
    total_reqs = 0.0

    for proxy in executor_proxies:
        stats = proxy.get_stats()
        average_latency = ((average_latency * total_reqs) +
                           (stats[1] * stats[0])) / (total_reqs + stats[0])
        average_accuracy = ((average_accuracy * total_reqs) +
                            (stats[2] * stats[0])) / (total_reqs + stats[0])
        total_reqs += stats[0]

    print("Number of Requests: " + str(total_reqs))
    print("Average Latency:    " + str(average_latency))
    print("Accuracy:           " + str(average_accuracy))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 -m src.pipeline_entrypoint <system config path> <application config path> <data config path> <debug arguments>")
        exit()

    system_config_file = sys.argv[1]
    application_config_file = sys.argv[2]
    data_config_file = sys.argv[3]

    with open(system_config_file, "r") as fh:
        system_config: SystemConfig = json.load(fh, cls=SystemConfigDecoder)

    with open(application_config_file, "r") as fh:
        application_config: ApplicationConfig = json.load(
            fh, cls=ApplicationConfigDecoder)

    with open(data_config_file, "r") as fh:
        data_config: DataConfig = json.load(fh, cls=DataConfigDecoder)

    if '--debug' in sys.argv:
        for app in application_config.applications:
            mods = app.model_profiles
            print(f"APPLICATION: {app.name}")
            print(
                "NAME            AVG UTIL        NEGATIVE        POSITIVE        ACCURACY (F1)")
            print(
                "-----------------------------------------------------------------------------")
            for model in mods:
                print(model.name.ljust(12), "  ", str(model.expected_utility().__round__(6)).ljust(12), "  ",
                      str(model.expected_conditional_utility(
                          [1.0, 0.0]).__round__(6)).ljust(12), "  ",
                      str(model.expected_conditional_utility(
                          [0.0, 1.0]).__round__(6)).ljust(12), "  ",
                      str(model.get_inference_accuracy().__round__(6)).ljust(12))

    if '--data-aware' in sys.argv:
        system_config.data_aware = True

    run_pipeline(system_config, application_config, data_config)
