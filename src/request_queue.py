import multiprocessing
from typing import Dict
from multiprocessing.queues import Queue
from .request import InferenceRequest


class SharedValue(object):

    def __init__(self, x=0.0):
        self.value = multiprocessing.Value('d', x)

    @property
    def get(self):
        return self.value.value

    def add(self, x):
        with self.value.get_lock():
            self.value.value += x

    def subtract(self, x):
        with self.value.get_lock():
            self.value.value -= x


class RequestQueue(Queue):

    def __init__(self, retrieval_latencies: Dict):
        super().__init__(ctx=multiprocessing.get_context())
        self._latency = SharedValue(0.0)
        self._retrieval_latencies = retrieval_latencies

    def __getstate__(self):
        return {
            'parent_state': super().__getstate__(),
            '_latency': self._latency,
        }

    def __setstate__(self, state):
        super().__setstate__(state['parent_state'])
        self._latency = state['_latency']

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        req: InferenceRequest = args[0]
        if isinstance(req, InferenceRequest):
            mlat = req.get_expected_processing_latency(
                self._retrieval_latencies)
            self._latency.add(mlat)

    def get(self, *args, **kwargs):
        item: InferenceRequest = super().get(*args, **kwargs)
        if isinstance(item, InferenceRequest):
            mlat = item.get_expected_processing_latency(
                self._retrieval_latencies)
            self._latency.subtract(mlat)
        return item

    def get_queue_latency(self):
        return self._latency.get
