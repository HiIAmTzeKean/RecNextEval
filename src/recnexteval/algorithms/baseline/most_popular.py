import logging
from typing import Self

import numpy as np
from scipy.sparse import csr_matrix

from ...matrix import PredictionMatrix
from ..core import IncrementalTrainingMixin, PopularityPaddingMixin, TopKAlgorithm


logger = logging.getLogger(__name__)


class MostPopular(TopKAlgorithm, PopularityPaddingMixin, IncrementalTrainingMixin):
    """A popularity-based algorithm that considers all historical data."""

    IS_BASE: bool = False
    X_: csr_matrix | None = None  # Store all historical training data

    def _fit(self, X: csr_matrix) -> Self:
        if self.X_ is not None:
            self._append_training_data(X)
        else:
            self.X_ = X.copy()

        if not isinstance(self.X_, csr_matrix):
            raise ValueError("Training data is not initialized properly.")

        if self.X_.shape[1] < self.K:
            logger.warning("K is larger than the number of items.", UserWarning)

        self.sorted_scores_ = self.get_popularity_scores(self.X_)
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        intended_shape = (X.filter_for_predict().num_interactions, X.user_item_shape[1])

        # Vectorized: repeat the sorted scores for each prediction row
        data = np.tile(self.sorted_scores_, (intended_shape[0], 1))
        X_pred = csr_matrix(data)

        return X_pred
