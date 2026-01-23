import logging

from ...matrix import InteractionMatrix
from ..base import Setting
from ..splitters import (
    NLastInteractionSplitter,
)


logger = logging.getLogger(__name__)


class LeaveNOutSetting(Setting):
    """Leave-N-Out setting for splitting data.

    Splits the dataset into training and test sets by leaving out the last N interactions
    for each user as test data, using the previous n_seq_data interactions as context.
    """

    IS_BASE: bool = False

    def __init__(
        self,
        n_seq_data: int = 1,
        N: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.n_seq_data = n_seq_data
        # we use top_K to denote the number of items to predict
        self.top_K = N
        logger.info("Splitting data")
        self._splitter = NLastInteractionSplitter(N, n_seq_data)

    def _split(self, data: InteractionMatrix) -> None:
        """Splits the dataset into training and test sets based on interaction timestamps.

        Args:
            data: Interaction matrix to be split. Must contain timestamps.
        """

        self._training_data, future_interaction = self._splitter.split(data)
        # we need to copy the data to avoid modifying the background data
        past_interaction = self._training_data.copy()

        self._unlabeled_data, self._ground_truth_data = self.prediction_data_processor.process(
            past_interaction=past_interaction,
            future_interaction=future_interaction,
            top_K=self.top_K,
        )
        self._t_window = None
