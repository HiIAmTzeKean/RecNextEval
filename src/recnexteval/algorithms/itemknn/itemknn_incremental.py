import logging
from typing import Self

from scipy.sparse import csr_matrix

from ..core import IncrementalTrainingMixin, PopularityPaddingMixin, TopKItemSimilarityMatrixAlgorithm
from .itemknn import ItemKNN


logger = logging.getLogger(__name__)


class ItemKNNIncremental(ItemKNN, IncrementalTrainingMixin):
    """Incremental version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    IS_BASE: bool = False

    def __init__(self, K: int = 10, pad_with_popularity: bool = True) -> None:
        PopularityPaddingMixin.__init__(self, pad_with_popularity=pad_with_popularity)
        TopKItemSimilarityMatrixAlgorithm.__init__(self, K=K)
        self.X_: None | csr_matrix = None

    def _fit(self, X: csr_matrix) -> Self:
        """Fit a cosine similarity matrix from item to item."""
        if self.X_ is not None:
            self._append_training_data(X)
            super()._fit(self.X_)
        else:
            super()._fit(X)
        return self
