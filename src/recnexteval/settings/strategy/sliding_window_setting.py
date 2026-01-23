import logging
from warnings import warn

import numpy as np
from tqdm import tqdm

from ...matrix import InteractionMatrix, TimestampAttributeMissingError
from ..base import Setting
from ..splitters import NLastInteractionTimestampSplitter, TimestampSplitter


logger = logging.getLogger(__name__)


class SlidingWindowSetting(Setting):
    """Sliding window setting for splitting data.

    The data is split into a background set and evaluation set. The evaluation set is defined by a sliding window
    that moves over the data. The window size is defined by the window_size parameter. The evaluation set comprises of the
    unlabeled data and ground truth data stored in a list. The unlabeled data contains the last n_seq_data interactions
    of the users/item before the split point along with masked interactions after the split point. The number of
    interactions per user/item is limited to top_K.
    The ground truth data is the interactions after the split point and spans window_size seconds.

    Args:
        background_t (int): Time point to split the data into background and evaluation data. Split will be from [0, t).
        window_size (int, optional): Size of the window in seconds to slide over the data.
            Affects the incremental data being released to the model. If
            t_ground_truth_window is not provided, ground truth data will also
            take this window. Defaults to np.iinfo(np.int32).max.
        n_seq_data (int, optional): Number of last sequential interactions to provide as
             data for model to make prediction. Defaults to 0.
        top_K (int, optional): Number of interaction per user that should be selected for evaluation purposes.
            Defaults to 10.
        t_upper (int, optional): Upper bound on the timestamp of interactions.
            Defaults to maximal integer value (acting as infinity).
        t_ground_truth_window (int, optional): Size of the window in seconds to slide over the data for ground truth data.
            If not provided, defaults to window_size during computation.
        seed (int, optional): Seed for random number generator. Defaults to 42.
    """

    IS_BASE: bool = False

    def __init__(
        self,
        background_t: int,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        n_seq_data: int = 0,
        top_K: int = 10,
        t_upper: int = np.iinfo(np.int32).max,
        t_ground_truth_window: None | int = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self._sliding_window_setting = True
        self.t = background_t
        self.window_size = window_size
        """Window size in seconds for splitter to slide over the data."""
        self.n_seq_data = n_seq_data
        self.top_K = top_K
        self.t_upper = t_upper
        """Upper bound on the timestamp of interactions. Defaults to maximal integer value (acting as infinity)."""

        if t_upper and t_upper < background_t:
            raise ValueError("t_upper must be greater than background_t")

        if t_ground_truth_window is None:
            t_ground_truth_window = window_size

        self.t_ground_truth_window = t_ground_truth_window

        self._background_splitter = TimestampSplitter(t=background_t, t_lower=None, t_upper=self.t_upper)
        self._window_splitter = NLastInteractionTimestampSplitter(
            t=background_t,
            t_upper=t_ground_truth_window,
            n_seq_data=n_seq_data,
        )

    def _split(self, data: InteractionMatrix) -> None:
        if not data.has_timestamps:
            raise TimestampAttributeMissingError()
        if data.min_timestamp > self.t:
            warn(
                f"Splitting at time {self.t} is before the first "
                "timestamp in the data. No data will be in the background(training) set."
            )
        if self.t_upper:
            data = data.timestamps_lt(self.t_upper)

        self._training_data, _ = self._background_splitter.split(data)
        self._ground_truth_data, self._unlabeled_data, self._t_window, self._incremental_data = (
            [],
            [],
            [],
            [],
        )

        # sub_time is the subjugate time point that the splitter will slide over the data
        sub_time = self.t
        max_timestamp = data.max_timestamp

        pbar = tqdm(total=int((max_timestamp - sub_time) / self.window_size))
        while sub_time <= max_timestamp:
            self._t_window.append(sub_time)
            # the set used for eval will always have a timestamp greater than
            # data released such that it is unknown to the model
            self._window_splitter.update_split_point(t=sub_time)
            past_interaction, future_interaction = self._window_splitter.split(data)

            # if past_interaction, future_interaction is empty, log an info message
            if len(past_interaction) == 0:
                logger.info(
                    "Split at time %s resulted in empty unlabelled testing samples.", sub_time
                )
            if len(future_interaction) == 0:
                logger.info("Split at time %s resulted in empty incremental data.", sub_time)

            unlabeled_set, ground_truth = self.prediction_data_processor.process(
                past_interaction=past_interaction,
                future_interaction=future_interaction,
                top_K=self.top_K,
            )
            self._unlabeled_data.append(unlabeled_set)
            self._ground_truth_data.append(ground_truth)

            self._incremental_data.append(future_interaction)

            sub_time += self.window_size
            pbar.update(1)
        pbar.close()

        self._num_split_set = len(self._unlabeled_data)
        logger.info(
            "Finished split with window size %s seconds. Number of splits: %s in total.",
            self.window_size,
            self._num_split_set,
        )
