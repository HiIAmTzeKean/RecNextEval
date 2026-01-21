import logging
from dataclasses import dataclass
from warnings import warn

from tqdm import tqdm

from ...matrix import PredictionMatrix
from ...registries import METRIC_REGISTRY, MetricEntry
from ...settings import EOWSettingError, Setting
from ..accumulator import MetricAccumulator
from ..base import EvaluatorBase
from ..state_management import AlgorithmStateManager


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EvaluatorPipeline(EvaluatorBase):
    """Evaluation via pipeline.

    The diagram below shows the diagram of the pipeline evaluator for the
    sliding window setting. If the setting is a single time point setting, the
    evaluator will only run phase 1 and 2. This is the same as setting the sliding
    window setting to only having 1 split.

    .. image:: /_static/pipeline_scheme.png
        :align: center
        :scale: 40 %
        :alt: Pipeline diagram

    The pipeline is responsible for evaluating algorithms with metrics and evaluates
    the algorithms in 3 phases:

    1. Pre-phase
    2. Evaluation phase
    3. Data release phase

    In Setting 3 (single time point split), the evaluator will only run phase 1 and 2.
    In Setting 1 (sliding window setting), the evaluator will run all 3 phases, with
    phase 2 and 3 repeated for each window/split. This can be seen in the diagram
    above.
    """

    algo_state_mgr: AlgorithmStateManager

    def _ready_algo(self) -> None:
        """Train the algorithms with the background data.

        This method should be called after algorithm instantiation. The
        algorithms are trained with the background data, and the set of known
        user/item is updated.

        Raises:
            ValueError: If algorithm is not instantiated.
        """
        training_data = self.setting.training_data
        self.user_item_base.update_known_user_item_base(training_data)
        training_data = PredictionMatrix.from_interaction_matrix(training_data)
        training_data.mask_user_item_shape(self.user_item_base.known_shape)

        for algo_state in self.algo_state_mgr.values():
            algo_state.algorithm_ptr.fit(X=training_data)

    def _ready_evaluator(self) -> None:
        """Pre-phase of the evaluator (Phase 1).

        This method prepares the evaluator for the evaluation process.
        It instantiates the algorithm, trains the algorithm with the background data,
        instantiates the metric accumulator, and prepares the data generators.
        The next phase of the evaluator is the evaluation phase (Phase 2).

        The evaluator is prepared by:
        1. Instantiating each algorithm from the algorithm entries
        2. For each algorithm, train the algorithm with the background data from
           the setting
        3. Instantiate the metric accumulator for micro and macro metrics
        4. Create an entry for each metric in the macro metric accumulator
        5. Prepare the data generators for the setting
        """
        logger.info("Phase 1: Preparing the evaluator...")
        self._ready_algo()
        logger.debug("Algorithms trained with background data...")

        self._acc = MetricAccumulator()
        logger.debug("Metric accumulator instantiated...")

        self.setting.restore()
        logger.debug("Setting data generators ready...")

    def _evaluate_step(self) -> None:
        """Evaluate performance of the algorithms (Phase 2).

        This method evaluates the performance of the algorithms with the metrics.
        It takes the unlabeled data, predicts the interaction, and evaluates the
        performance with the ground truth data.

        The evaluation is done by:
        1. Get the next unlabeled data and ground truth data from the setting
        2. Get the current timestamp from the setting
        3. Update the unknown user/item base with the ground truth data
        4. Mask the unlabeled data to the known user/item base shape
        5. Mask the ground truth data based on the provided flags for ignoring
           unknown users and items
        6. Predict the interaction with the algorithms
        7. Check the shape of the prediction matrix
        8. Store the results in the micro metric accumulator
        9. Cache the results in the macro metric accumulator
        10. Repeat step 6 for each algorithm

        Raises:
            EOWSettingError: If there is no more data to be processed.
        """
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
        """Data release phase (Phase 3).

        This method releases the data from the evaluator. This method should only
        be called when the setting is a sliding window setting.

        The data is released by updating the known user/item base with the
        incremental data. After updating the known user/item base, the incremental
        data is masked to the known user/item base shape. The algorithms are then
        trained with the incremental data.

        Note:
            Previously unknown user/item base is reset after the data release.
            Since these unknown user/item base should be within the incremental
            data released.
        """
        if not self.setting.is_sliding_window_setting:
            return
        logger.info("Phase 3: Releasing the data...")

        incremental_data = self._get_training_data()
        for algo_state in self.algo_state_mgr.values():
            algo_state.algorithm_ptr.fit(incremental_data)

    def run_step(self, reset=False) -> None:
        """Run a single step of the evaluator.

        This method runs a single step of the evaluator. The evaluator is split
        into 3 phases. In the first run, all 3 phases are run. In the subsequent
        runs, only the evaluation and data release phase are run. The method
        will run all steps until the number of splits is reached. To rerun the
        evaluation again, call with reset=True.

        Args:
            reset: Whether to reset the evaluation step. Defaults to False.
        """
        if reset:
            self._run_step = 0
            logger.info(f"There is a total of {self.setting.num_split} steps. Running step {self._run_step}")
            self._ready_evaluator()
            logger.info("Evaluator ready...")
            self._evaluate_step()
            self._data_release_step()
            return

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

        Runs all 3 phases across all splits (if there are multiple splits).
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
