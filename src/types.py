import random
import string

from typing import List, Set, Dict, Tuple
import numpy as np
from statistics import mean

class Message:

    def __init__(self) -> None:
        self._token = self._gen_token()

    @property
    def token(self) -> str:
        return self._token

    @classmethod
    def _gen_token(cls) -> str:
        return ''.join(
            random.choice(string.ascii_uppercase + string.digits)
            for _ in range(8)
        )

    def get_expected_processing_latency(self):
        raise NotImplementedError()


# NOTE: utility_matrix is not a property of the model, but of the application.
# However, it was easiest to just pass the utility_matrix from the application
# to the model object, as all ModelInstance objects are associated with
# one Application only.

class ModelInstance:
    def __init__(
        self,
        name,
        task,
        use_profile,
        modality,
        utility_matrix,
        confusion_matrix,
        latencies,
        impl,
    ) -> None:
        self.name: str = name
        self.task: str = task
        self.use_profile: bool = use_profile
        self.modality: str = modality
        self.utility_matrix: List[List[float]] = utility_matrix
        self.confusion_matrix: List[List[float]] = confusion_matrix
        self.latencies: Dict[str: Tuple[float, float]] = latencies
        self.impl: Dict[str, str] = impl

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return f"{self.name} : {self.expected_utility()}"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash((self.name, self.task))

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def expected_utility(self) -> float:
        # return average_utility(self._cmat, self.task.utility_matrix)
        return self.get_inference_accuracy()

    def expected_conditional_utility(self, class_probs) -> float:
        # return data_aware_utility(self._cmat, self.task.utility_matrix,
        # class_probs)
        return self.expected_conditional_accuracy(class_probs)

    def true_utility(self, y_true: int, y_hat: int) -> float:
        return self.utility_matrix[y_true][y_hat]

    def get_inference_accuracy(self) -> float:
        cm = self._convert_matrix_lists_to_numpy(self.confusion_matrix)
        return np.diag(cm).sum() / cm.sum()

    def expected_conditional_accuracy(self, class_probs) -> float:
        cm = self._convert_matrix_lists_to_numpy(self.confusion_matrix)
        class_scores = np.diag(cm) / cm.sum(1)
        return np.array(class_probs).dot(class_scores)

    def get_latency(self, last_model) -> float:
        latency = self.get_inference_latency()
        return (
            latency + self.get_context_switch_latency()
            if last_model is None or last_model != self
            else latency
        )

    def get_inference_latency(self, server_name: str | None = None) -> float:
        if server_name is None:
            return mean(latency[0] for latency in self.latencies.values())
        return self.latencies[server_name][0]

    def get_context_switch_latency(
            self, server_name: str | None = None) -> float:
        if server_name is None:
            return mean(latency[1] for latency in self.latencies.values())
        return self.latencies[server_name][1]

    def _convert_matrix_lists_to_numpy(self, lists : List):
        d = len(lists[0])
        return np.array(lists, dtype=float).reshape((d, d))

# TODO: Investigate utility calculation code organization. Should this be
# more owned by the scheduler implementation?


def average_utility(
    confusion_mat: List[List[float]],
    utility_mat: List[List[float]],
) -> float:
    """
    Computes the average expected utility of a model (conditioned on the training data).
    This is somewhat more sophisticated version of approaches in the literature which make
    decisions based on the average model accuracy.

    NOTE: We are just assuming 2x2 for now.
    """
    denom = sum([sum(r) for r in confusion_mat])

    tn = confusion_mat[0][0] / denom
    tp = confusion_mat[1][1] / denom
    fp = confusion_mat[0][1] / denom
    fn = confusion_mat[1][0] / denom
    return (
        tn * utility_mat[0][0] + tp * utility_mat[1][1] +
        fp * utility_mat[0][1] + fn * utility_mat[1][0]
    )


def data_aware_utility(
    confusion_mat: List[List[float]],
    utility_mat: List[List[float]],
    class_probs: List[float],
) -> float:
    """
    Estimates the utility of a model using data-awareness.
    The specified class probabilities are used to weight the utility function.

    NOTE: Just assuming 2x2 for now.
    """
    # If label is actually negative (0)
    denom = sum(confusion_mat[0])
    true_negative_util = (
        (confusion_mat[0][0] / denom) * utility_mat[0][0] +
        (confusion_mat[0][1] / denom) * utility_mat[0][1]
    )

    # If label is actually positive (1)
    denom = sum(confusion_mat[1])
    true_positive_util = (
        (confusion_mat[1][0] / denom) * utility_mat[1][0] +
        (confusion_mat[1][1] / denom) * utility_mat[1][1]
    )
    return (
        class_probs[0] * true_negative_util +
        class_probs[1] * true_positive_util
    )

# This class encompasses all data required to represent a specific application to be
# used within the framework.
# Applications may have multiple different models associated with them that may
# be expecting data of different modalities, however, the SLA latency and utility matrix
# for an application will be the same across each of the data modalities.


class Application:
    def __init__(self, name: str, sla_latency: float, sample_duration: int,
                 utility_matrix: List[List[float]], prior: List[float], model_profiles: List[ModelInstance]):
        self.name = name
        self.sla_latency = sla_latency
        self.sample_duration = sample_duration
        self.utility_matrix = utility_matrix
        self.prior = prior
        self.model_profiles = model_profiles

    @classmethod
    def from_dict(cls, data):
        name = data["name"]
        sla_latency = data["sla_latency"]
        sample_duration = data["sample_duration"]
        utility_matrix = data["utility_matrix"]
        prior = data["prior"]
        model_profiles = [ModelInstance.from_dict(
            profile) for profile in data["model_profiles"]]
        return Application(name, sla_latency, sample_duration,
                           utility_matrix, prior, model_profiles)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    @property
    def num_classes(self) -> int:
        return len(self.utility_matrix)

    def get_target_latency(self) -> float:
        return self.sla_latency

    def get_sample_duration(self) -> float:
        return self.sample_duration

    def get_supported_modalities(self) -> Set[str]:
        modalities = set()
        for model in self.model_profiles:
            modalities.add(model.modality)
        return modalities

    def get_models(self) -> List[ModelInstance]:
        return self.model_profiles

    def __repr__(self):
        return (f"Application(name={self.name}, sla_latency={self.sla_latency}, "
                f"sample_duration={self.sample_duration}, utility_matrix={self.utility_matrix}, "
                f"prior={self.prior}, model_profiles={self.model_profiles})")
