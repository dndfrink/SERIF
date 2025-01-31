
#from importlib.metadata import version

#__version__ = "1.0.0"

from typing import List, Optional
import numpy as np
# import faiss

from src.types import Application 

class SneakPeekModel:

    def __init__(
        self,
        app: Application,
        threshold: float = 0.4,
    ) -> None:
        self._index = None
        self._app = app 
        self._prior = app.prior
        self._y = None
        self._threshold = threshold

    def fit(self, X, y) -> None:
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X.astype(np.float32))
        self._y = y

    # Returns class probabilities
    def infer(self, X, k: int = 5) -> List[float]:
        distances, indices = self._index.search(X.astype(np.float32), k=k)
        votes = self._y[indices][0]
        probas = [np.bincount(votes, minlength=2) / float(len(votes))]
        evidence = [0.0, 1.0] if probas[1] > self._threshold else [1.0, 0.0]

        return self._get_expected_dirichlet_posterior(evidence)

    def _get_expected_dirichlet_posterior(
        self,
        multinomial_evidence: List[int],
    ) -> List[float]:
        post_parms = [
            float(multinomial_evidence[i]) + self._prior[i]
            for i in range(len(self._prior))
        ]
        alpha_0 = sum(post_parms)
        return [ p / alpha_0 for p in post_parms ]
