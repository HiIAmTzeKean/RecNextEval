import logging
from typing import Self

import numpy as np
from scipy.sparse import csr_matrix

from ...matrix import PredictionMatrix
from ..core import IncrementalTrainingMixin, PopularityPaddingMixin, TopKAlgorithm


logger = logging.getLogger(__name__)


class DecayPopularity(TopKAlgorithm, PopularityPaddingMixin, IncrementalTrainingMixin):
    IS_BASE: bool = False
    X_: csr_matrix | None = None
    decay_type = "exponential"
    decay_rate = 0.01
    internal_clock = 0

    def _compute_decay_weights(self, time_delta):
        """Apply decay based on time."""
        if self.decay_type == "exponential":
            return np.exp(-self.decay_rate * time_delta)
        elif self.decay_type == "linear":
            return np.clip(1 - self.decay_rate * time_delta, 0, 1)
        else:
            return 1.0 / (1.0 + self.decay_rate * (time_delta**2))

    def _fit(self, X: csr_matrix) -> Self:
        if self.X_ is not None:
            self._append_training_data(X)
        else:
            self.X_ = X.copy()

        if not isinstance(self.X_, csr_matrix):
            raise ValueError("Training data is not initialized properly.")

        if self.X_.shape[1] < self.K:
            logger.warning("K is larger than the number of items.", UserWarning)

        # Compute decay weights
        decay_weights = self._compute_decay_weights(self.internal_clock)
        self.X_.data = self.X_.data * decay_weights

        # Get top-K items
        self.sorted_scores_ = self.get_popularity_scores(self.X_)
        self.internal_clock += 1
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        intended_shape = (X.filter_for_predict().num_interactions, X.user_item_shape[1])

        # Vectorized: repeat the sorted scores for each prediction row
        data = np.tile(self.sorted_scores_, (intended_shape[0], 1))
        X_pred = csr_matrix(data)

        return X_pred
