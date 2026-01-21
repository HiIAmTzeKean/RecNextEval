import logging
from dataclasses import dataclass
from warnings import warn

from tqdm import tqdm

from ...matrix import PredictionMatrix
from ...registries import METRIC_REGISTRY
from ...settings import EOWSettingError
from ..accumulator import MetricAccumulator
from ..base import EvaluatorBase
from ..state_management import AlgorithmStateManager


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EvaluatorPipeline(EvaluatorBase):
    """Evaluation via pipeline."""

    algo_state_mgr: AlgorithmStateManager

    def _ready_algo(self) -> None:
        training_data = self._get_training_data()

        for algo_state in self.algo_state_mgr.values():
            algo_state.algorithm_ptr.fit(X=training_data)

    def _ready_evaluator(self) -> None:
        logger.info("Phase 1: Preparing the evaluator...")
        self._ready_algo()
        logger.debug("Algorithms trained with background data...")

        self._acc = MetricAccumulator()
        logger.debug("Metric accumulator instantiated...")

        self.setting.restore()
        logger.debug("Setting data generators ready...")

    def _evaluate_step(self) -> None:
        logger.info("Phase 2: Evaluating the algorithms...")
        try:
            unlabeled_data, ground_truth_data, current_timestamp = self._get_evaluation_data()
        except EOWSettingError as e:
            raise e

        # get the top k interaction per user
        # X_true = ground_truth_data.get_users_n_first_interaction(self.metric_k)
        X_true = ground_truth_data.item_interaction_sequence_matrix
        for algo_state in self.algo_state_mgr.values():
            X_pred = algo_state.algorithm_ptr.predict(unlabeled_data)
            logger.debug("Shape of prediction matrix: %s", X_pred.shape)
            logger.debug("Shape of ground truth matrix: %s", X_true.shape)
            X_pred = self._prediction_unknown_item_handler(X_true, X_pred)

            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                params = {
                    'timestamp_limit': current_timestamp,
                    'user_id_sequence_array': ground_truth_data.user_id_sequence_array,
                    'user_item_shape': ground_truth_data.user_item_shape,
                }
                if metric_entry.K is not None:
                    params['K'] = metric_entry.K
                metric = metric_cls(**params)
                metric.calculate(X_true, X_pred)
                self._acc.add(
                    metric=metric,
                    algorithm_name=self.algo_state_mgr.get_algorithm_identifier(algo_state.algorithm_uuid),
                )

    def _data_release_step(self) -> None:
        if not self.setting.is_sliding_window_setting:
            return
        logger.info("Phase 3: Releasing the data...")

        incremental_data = self._get_training_data()
        for algo_state in self.algo_state_mgr.values():
            algo_state.algorithm_ptr.fit(incremental_data)

    def reset(self) -> None:
        """Reset the evaluator to initial state."""
        logger.info("Resetting the evaluator for a new run...")
        self._run_step = 0

    def run_step(self) -> None:
        """Run a single step of the evaluator."""
        if self._run_step == 0:
            logger.info(f"There is a total of {self.setting.num_split} steps. Running step {self._run_step}")
            self._ready_evaluator()

        if self._run_step > self.setting.num_split:
            logger.info("Finished running all steps, call `run_step(reset=True)` to run the evaluation again")
            warn("Running this method again will not have any effect.")
            return
        logger.info("Running step %d", self._run_step)
        self._evaluate_step()
        self._data_release_step()

    def run_steps(self, num_steps: int) -> None:
        """Run multiple steps of the evaluator.

        Effectively runs the run_step method num_steps times. Call
        this method to run multiple steps of the evaluator at once.

        Args:
            num_steps: Number of steps to run.

        Raises:
            ValueError: If cannot run the specified number of steps.
        """
        if self._run_step + num_steps > self.setting.num_split:
            raise ValueError(f"Cannot run {num_steps} steps, only {self.setting.num_split - self._run_step} steps left")
        for _ in tqdm(range(num_steps)):
            self.run_step()

    def run(self) -> None:
        """Run the evaluator across all steps and splits.

        This method should be called when the programmer wants to step through
        all phases and splits to arrive to the metrics computed. An alternative
        to running through all splits is to call the run_step method which runs
        only one step at a time.
        """
        self._ready_evaluator()

        with tqdm(total=self.setting.num_split, desc="Evaluating steps") as pbar:
            while self._run_step <= self.setting.num_split:
                logger.info("Running step %d", self._run_step)
                self._evaluate_step()
                pbar.update(1)
                # if is last step, no need to release data anymore
                # since there is no more evaluation that can be done
                # break out of the loop
                if self._run_step == self.setting.num_split:
                    break
                self._data_release_step()
