import datetime as dt
from typing import Dict

from src.types import Application, Message, ModelInstance

class TerminationRequest(Message):

    def __init__(self):
        super().__init__()

    def get_expected_processing_latency(self):
        raise NotImplementedError()


class InferenceRequest(Message):

    def __init__(self, app: Application) -> None:

        self._application = app
        self._assigned_model: ModelInstance = None
        self._modalities_cached = {}
        self._assigned_server: str = None

        self.accuracy = 0.0
        self.confidence = 0.0

        self._true_label = -1
        self._predicted_label = -1
        self._class_probs = []

        self._arrival = None
        self._deadline = None
        self._complete_time = None

        super().__init__()

    @classmethod
    def from_dict(cls, data):
        req = InferenceRequest(Application.from_dict(data['_application']))
        try:
            req._assigned_model = ModelInstance.from_dict(
                data['_assigned_model'])
        except TypeError:
            req._assigned_model = None
        req._modalities_cached = data['_modalities_cached']
        req._assigned_server = data['_assigned_server']
        req.accuracy = data['accuracy']
        req.confidence = data['confidence']
        req._true_label = data['_true_label']
        req._predicted_label = data['_predicted_label']
        req._class_probs = data['_class_probs']
        req._arrival = data['_arrival']
        req._deadline = data['_deadline']
        req._complete_time = data['_complete_time']
        req._token = data['_token']
        return req

    @property
    def task(self):
        return self._application

    @property
    def assigned_model(self):
        return self._assigned_model

    @assigned_model.setter
    def assigned_model(self, model):
        self._assigned_model = model

    @property
    def assigned_server(self):
        return self._assigned_server

    @assigned_server.setter
    def assigned_server(self, server):
        self._assigned_server = server

    @property
    def class_probs(self):
        return self._class_probs

    @class_probs.setter
    def class_probs(self, probs):
        self._class_probs = probs

    @property
    def true_label(self):
        return self._true_label

    @true_label.setter
    def true_label(self, label):
        self._true_label = label

    @property
    def predicted_label(self):
        return self._predicted_label

    @predicted_label.setter
    def predicted_label(self, label):
        self._predicted_label = label

    @property
    def arrival(self):
        return dt.datetime.fromtimestamp(self._arrival)

    @property
    def complete_time(self):
        return dt.datetime.fromtimestamp(self._complete_time)

    @property
    def deadline(self):
        return dt.datetime.fromtimestamp(self._deadline)

    def set_modality_cached(self, modality: str, server_ip: str):
        self._modalities_cached[modality] = server_ip

    def target_latency(self) -> float:
        return self.task.get_target_latency()

    def start_the_clock(self):
        self._arrival = dt.datetime.utcnow().timestamp()
        self._deadline = (
            self.arrival +
            dt.timedelta(
                0,
                self.target_latency())).timestamp()

    def get_confidence(self):
        return self.confidence

    def set_confidence(self, conf):
        self.confidence = conf

    def get_accuracy(self):
        return self.accuracy

    def over_latency_target(self):
        return self.complete_time > self.get_deadline()

    def get_arrival_time(self):
        return dt.datetime.fromtimestamp(self.arrival)

    def get_remaining_time(self):
        curTime = dt.datetime.utcnow()
        if curTime < self.deadline:
            return (self.deadline - curTime).total_seconds()
        return 0.0

    def get_deadline(self):
        return self.deadline

    def is_complete(self):
        return self._complete_time is not None

    def mark_complete(self):
        self._complete_time = dt.datetime.utcnow().timestamp()

    def get_retrieval_latency(self, retrieval_latencies: Dict) -> float:
        if self._modalities_cached.get(
                self._assigned_model.modality) == self.assigned_server:
            return 0.0
        else:
            return retrieval_latencies[self._assigned_model.modality][self.assigned_server]

    def get_duration(self):
        if self.is_complete():
            return (self.complete_time - self.arrival).total_seconds()
        return (dt.datetime.utcnow() - self.arrival).total_seconds()

    def set_accuracy(self, acc):
        self.accuracy = acc

    def get_expected_processing_latency(self, retrieval_latencies: Dict):
        # nit, context switch or no?
        # Could get this data from the queue if we need it.
        # Adding a 20% penalty for now
        return (self.assigned_model.get_inference_latency(
            self.assigned_server) * 1.2) + self.get_retrieval_latency(retrieval_latencies)

    def dump(self, verbose):

        print("Request " + self.token + ":")
        print("  Duration:        " + str(self.get_duration()))

        if verbose:
            print("  Model:           " + str(self._assigned_model))
            print("  True Label:      " + str(self._true_label))
            print("  Predicted Label: " + str(self.predicted_label))
            print("  Arrival:         " + str(self.arrival))

            if self.complete_time > self.get_deadline():
                miss = self.complete_time - self.get_deadline()
                print("  SLA MISS:        " + str(miss))
