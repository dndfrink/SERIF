from dataclasses import dataclass
from typing import List

@dataclass
class SchedulerNode:
    ip_address: str
    port: str
    url: str

    def __init__(self, ip_address: str, port: str):
        self.ip_address = ip_address
        self.port = port
        self.url = f"http://{self.ip_address}:{self.port}"

class RedisNode:
    def __init__(self, ip_address: str, ports: List[str]):
        self.ip_address = ip_address
        self.ports = ports

    def __repr__(self) -> str:
        return (f"RedisNode(ip_address={self.ip_address}, ports={self.ports})")


class ExecutorNode:
    def __init__(self, name: str, ip_address: str, port: str) -> None:
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.url = f"http://{self.ip_address}:{self.port}"

    def __hash__(self):
        return hash((self.name, self.ip_address, self.port, "Executor"))

    def __repr__(self) -> str:
        return (f"ExecutorNode(name={self.name}, ip_address={self.ip_address}, port={self.port})")
