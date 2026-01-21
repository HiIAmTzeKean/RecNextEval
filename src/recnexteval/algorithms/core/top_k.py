
import logging

from scipy.sparse import csr_matrix

from .base import Algorithm


logger = logging.getLogger(__name__)


class TopKAlgorithm(Algorithm):
    """Base algorithm for algorithms that recommend top-K items for every user."""

    def __init__(self, K: int = 10) -> None:
        super().__init__()
        self.K = K


class TopKItemSimilarityMatrixAlgorithm(TopKAlgorithm):
    """Base algorithm for algorithms that fit an item to item similarity model with K similar items for every item

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.
    """

    similarity_matrix_: csr_matrix

    def _check_fit_complete(self) -> None:
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Ensures that similarity_matrix_ is computed
        if not hasattr(self, "similarity_matrix_"):
            raise AttributeError(f"{self.name} has no attribute similarity_matrix_ after fitting.")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            logger.warning(f"{self.name} missing similar items for {missing} items.")
