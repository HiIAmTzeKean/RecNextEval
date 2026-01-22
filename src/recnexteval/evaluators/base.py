import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from scipy.sparse import csr_matrix

from ..matrix import PredictionMatrix
from ..registries import METRIC_REGISTRY, MetricEntry
from ..settings import EOWSettingError, Setting
from .core import MetricAccumulator, MetricLevelEnum, UserItemKnowledgeBase


logger = logging.getLogger(__name__)


@dataclass
class EvaluatorBase:
    """Base class for evaluator.

    Provides the common methods and attributes for the evaluator classes. Should
    there be a need to create a new evaluator, it should inherit from this class.

    Args:
        metric_entries: List of metric entries to compute.
        setting: Setting object.
        metric_k: Value of K for the metrics.
        ignore_unknown_user: Ignore unknown users, defaults to False.
        ignore_unknown_item: Ignore unknown items, defaults to False.
        seed: Random seed for reproducibility.
    """

    metric_entries: list[MetricEntry]
    setting: Setting
    metric_k: int
    ignore_unknown_user: bool = False
    ignore_unknown_item: bool = False
    seed: int = 42
    user_item_base: UserItemKnowledgeBase = field(default_factory=UserItemKnowledgeBase)
    _run_step: int = 0
    _acc: MetricAccumulator = field(init=False)
    _current_timestamp: int = field(init=False)

    def _get_training_data(self) -> PredictionMatrix:
        if self._run_step == 0:
            logger.debug("First step, getting training data")
            training_data = self.setting.training_data
        else:
            logger.debug("Not first step, getting previous ground truth data as training data")
            training_data = self.setting.get_split_at(self._run_step).incremental
            if training_data is None:
                raise ValueError("Incremental data is None in sliding window setting")
            self.user_item_base.reset_unknown_user_item_base()
        self.user_item_base.update_known_user_item_base(training_data)
        training_data = PredictionMatrix.from_interaction_matrix(training_data)
        training_data.mask_user_item_shape(shape=self.user_item_base.known_shape)
        return training_data

    def _get_evaluation_data(self) -> tuple[PredictionMatrix, PredictionMatrix, int]:
        """Get the evaluation data for the current step.

        Internal method to get the evaluation data for the current step. The
        evaluation data consists of the unlabeled data, ground truth data, and
        the current timestamp which will be returned as a tuple. The shapes
        are masked based through `user_item_base`. The unknown users in
        the ground truth data are also updated in `user_item_base`.

        Note:
            `_current_timestamp` is updated with the current timestamp.

        Returns:
            Tuple of unlabeled data, ground truth data, and current timestamp.

        Raises:
            EOWSettingError: If there is no more data to be processed.
        """
        try:
            split = self.setting.get_split_at(self._run_step)
            unlabeled_data = split.unlabeled
            ground_truth_data = split.ground_truth
            if split.t_window is None:
                raise ValueError("Timestamp of the current split cannot be None")
            self._current_timestamp = split.t_window

            unlabeled_data = PredictionMatrix.from_interaction_matrix(unlabeled_data)
            ground_truth_data = PredictionMatrix.from_interaction_matrix(ground_truth_data)
            self._run_step += 1
        except EOWSettingError:
            raise EOWSettingError("There is no more data to be processed, EOW reached")

        self.user_item_base.update_unknown_user_item_base(ground_truth_data)
        mask_shape = self.user_item_base.global_shape

        if self.ignore_unknown_item:
            # get the unknown items from our knowledge base
            # drop all rows with unknown items in ground truth
            # drop corresponding rows in unlabeled data
            ground_truth_data = ground_truth_data.items_in(self.user_item_base.known_item)
            mask_shape = (mask_shape[0], self.user_item_base.known_shape[1])
        if self.ignore_unknown_user:
            # get the unknown users from our knowledge base
            # drop all columns with unknown users in ground truth
            # drop corresponding columns in unlabeled data
            ground_truth_data = ground_truth_data.users_in(self.user_item_base.known_user)
            mask_shape = (self.user_item_base.known_shape[0], mask_shape[1])
        unlabeled_data._df = unlabeled_data._df.loc[ground_truth_data._df.index]

        unlabeled_data.mask_user_item_shape(shape=mask_shape)
        ground_truth_data.mask_user_item_shape(shape=mask_shape)
        return unlabeled_data, ground_truth_data, self._current_timestamp

    def _add_metric_results_for_prediction(
            self,
            ground_truth_data: PredictionMatrix,
            y_pred: csr_matrix,
            algorithm_name: str,
        ) -> None:
        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            params = {
                'timestamp_limit': self._current_timestamp,
                'user_id_sequence_array': ground_truth_data.user_id_sequence_array,
                'user_item_shape': ground_truth_data.user_item_shape,
            }
            if metric_entry.K is not None:
                params['K'] = metric_entry.K

            metric = metric_cls(**params)
            metric.calculate(y_true=ground_truth_data.item_interaction_sequence_matrix, y_pred=y_pred)
            self._acc.add(metric=metric, algorithm_name=algorithm_name)

    def _prediction_unknown_item_handler(self, y_true: csr_matrix, y_pred: csr_matrix) -> csr_matrix:
        """Handle shape difference due to unknown items in ground truth matrix.

        Extends the number of columns in the prediction matrix to match the ground truth. This is equivalent
        to submitting zero prediction for unknown items.
        """
        if y_pred.shape[1] == y_true.shape[1]:
            return y_pred
        logger.warning(
            "Prediction matrix shape %s is different from ground truth matrix shape %s.",
            y_pred.shape,
            y_true.shape,
        )

        y_pred = csr_matrix(
            (y_pred.data, y_pred.indices, y_pred.indptr),
            shape=(y_pred.shape[0], y_true.shape[1]),
        )
        return y_pred

    def metric_results(
        self,
        level: MetricLevelEnum | Literal["macro", "micro", "window", "user"] = MetricLevelEnum.MACRO,
        only_current_timestamp: None | bool = False,
        filter_timestamp: None | int = None,
        filter_algo: None | str = None,
    ) -> pd.DataFrame:
        """Results of the metrics computed.

        Computes the metrics of all algorithms based on the level specified and
        return the results in a pandas DataFrame. The results can be filtered
        based on the algorithm name and the current timestamp.

        Specifics
        ---------
        - User level: User level metrics computed across all timestamps.
        - Window level: Window level metrics computed across all timestamps. This can
            be viewed as a macro level metric in the context of a single window, where
            the scores of each user is averaged within the window.
        - Macro level: Macro level metrics computed for entire timeline. This
            score is computed by averaging the scores of all windows, treating each
            window equally.
        - Micro level: Micro level metrics computed for entire timeline. This
            score is computed by averaging the scores of all users, treating each
            user and the timestamp the user is in as unique contribution to the
            overall score.

        Args:
            level: Level of the metric to compute, defaults to "macro".
            only_current_timestamp: Filter only the current timestamp, defaults to False.
            filter_timestamp: Timestamp value to filter on, defaults to None.
                If both `only_current_timestamp` and `filter_timestamp` are provided,
                `filter_timestamp` will be used.
            filter_algo: Algorithm name to filter on, defaults to None.

        Returns:
            Dataframe representation of the metric.
        """
        if isinstance(level, str) and not MetricLevelEnum.has_value(level):
            raise ValueError("Invalid level specified")
        level = MetricLevelEnum(level)

        if only_current_timestamp and filter_timestamp:
            raise ValueError("Cannot specify both only_current_timestamp and filter_timestamp.")

        timestamp = None
        if only_current_timestamp:
            timestamp = self._current_timestamp

        if filter_timestamp:
            timestamp = filter_timestamp

        return self._acc.df_metric(filter_algo=filter_algo, filter_timestamp=timestamp, level=level)

    def restore(self) -> None:
        """Restore the generators before pickling.

        This method is used to restore the generators after loading the object
        from a pickle file.
        """
        self.setting.restore(self._run_step)
        logger.debug("Generators restored")

    def current_step(self) -> int:
        """Return the current step of the evaluator.

        Returns:
            Current step of the evaluator.
        """
        return self._run_step
