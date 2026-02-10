import logging
from dataclasses import dataclass, field
from uuid import UUID

from scipy.sparse import csr_matrix

from ...algorithms import Algorithm
from ...matrix import InteractionMatrix, PredictionMatrix
from ...settings import EOWSettingError
from ..core import AlgorithmStateEnum, AlgorithmStateManager, EvaluatorBase, MetricAccumulator
from .state import EvaluatorState
from .strategy import EvaluationStrategy, SlidingWindowStrategy


logger = logging.getLogger(__name__)


@dataclass
class EvaluatorStreamer(EvaluatorBase):
    """Evaluation via streaming through API.

    The diagram below shows the diagram of the streamer evaluator for the
    sliding window setting. Instead of the pipeline, we allow the user to
    stream the data release to the algorithm. The data communication is shown
    between the evaluator and the algorithm. Note that while only 2 splits are
    shown here, the evaluator will continue to stream the data until the end
    of the setting where there are no more splits.

    ![stream scheme](/assets/_static/stream_scheme.png)

    This class exposes a few of the core API that allows the user to stream
    the evaluation process. The following API are exposed:

    1. :meth:`register_algorithm`
    2. :meth:`start_stream`
    3. :meth:`get_unlabeled_data`
    4. :meth:`submit_prediction`

    The programmer can take a look at the specific method for more details
    on the implementation of the API. The methods are designed with the
    methodological approach that the algorithm is decoupled from the
    the evaluating platform. And thus, the evaluator will only provide
    the necessary data to the algorithm and evaluate the prediction.

    Args:
        metric_entries: list of metric entries.
        setting: Setting object.
        metric_k: Number of top interactions to consider.
        ignore_unknown_user: To ignore unknown users.
        ignore_unknown_item: To ignore unknown items.
        seed: Random seed for the evaluator.
    """

    _strategy: EvaluationStrategy = field(init=False)
    _algo_state_mgr: AlgorithmStateManager = field(default_factory=AlgorithmStateManager)
    _unlabeled_data_cache: PredictionMatrix = field(init=False)
    _ground_truth_data_cache: PredictionMatrix = field(init=False)
    _training_data_cache: PredictionMatrix = field(init=False)
    _state: EvaluatorState = EvaluatorState.INITIALIZED

    def __post_init__(self) -> None:
        """Initialize fields that require computation."""
        self._strategy = SlidingWindowStrategy()

    @property
    def state(self) -> EvaluatorState:
        return self._state

    def _assert_state(self, expected: EvaluatorState | list[EvaluatorState], error_msg: str) -> None:
        """Assert evaluator is in expected state"""
        if not isinstance(expected, list):
            expected = [expected]

        if self._state not in expected:
            raise RuntimeError(f"{error_msg} (Current state: {self._state.value})")
        return

    def _transition_state(self, new_state: EvaluatorState, allow_from: list[EvaluatorState]) -> None:
        """Guard state transitions explicitly"""
        if self._state not in allow_from:
            raise ValueError(f"Cannot transition from {self._state} to {new_state}. Allowed from: {allow_from}")
        self._state = new_state
        logger.info(f"Evaluator transitioned to {new_state}")

    def start_stream(self) -> None:
        """Start the streaming process.

        Warning:
            Once `start_stream` is called, the evaluator cannot register any new algorithms.

        Raises:
            ValueError: If the stream has already started.
        """
        self._assert_state(expected=EvaluatorState.INITIALIZED, error_msg="Stream has already started")
        self.setting.restore()
        logger.debug("Preparing evaluator for streaming")
        self._acc = MetricAccumulator()
        self.load_next_window()
        logger.debug("Evaluator is ready for streaming")
        # TODO: allow programmer to register anytime
        self._transition_state(new_state=EvaluatorState.STARTED, allow_from=[EvaluatorState.INITIALIZED])

    def register_model(
        self,
        algorithm: Algorithm,
        algorithm_name: None | str = None,
    ) -> UUID:
        """Register the algorithm with the evaluator.

        This method is called to register the algorithm with the evaluator.
        The method will assign a unique identifier to the algorithm and store
        the algorithm in the registry.

        Warning:
            Once `start_stream` is called, the evaluator cannot register any new algorithms.
        """
        self._assert_state(EvaluatorState.INITIALIZED, "Cannot register algorithms after stream started")
        algo_id = self._algo_state_mgr.register(name=algorithm_name, algorithm_ptr=algorithm)
        logger.debug(f"Algorithm {algo_id} registered")
        return algo_id

    def get_algorithm_state(self, algorithm_id: UUID) -> AlgorithmStateEnum:
        """Get the state of the algorithm."""
        return self._algo_state_mgr[algorithm_id].state

    def get_all_algorithm_status(self) -> dict[str, AlgorithmStateEnum]:
        """Get the status of all algorithms."""
        return self._algo_state_mgr.all_algo_states()

    def load_next_window(self) -> None:
        self.user_item_base.reset_unknown_user_item_base()
        self._training_data_cache = self._get_training_data()
        self._unlabeled_data_cache, self._ground_truth_data_cache, self._current_timestamp = self._get_evaluation_data()
        self._algo_state_mgr.set_all_ready(data_segment=self._current_timestamp)

    def get_training_data(self, algo_id: UUID) -> InteractionMatrix:
        """Get training data for the algorithm.

        Args:
            algo_id: Unique identifier of the algorithm.

        Raises:
            ValueError: If the stream has not started.

        Returns:
            The training data for the algorithm.
        """
        self._assert_state(expected=[EvaluatorState.STARTED, EvaluatorState.IN_PROGRESS], error_msg="Call start_stream() first")

        logger.debug(f"Getting data for algorithm {algo_id}")

        if self._strategy.should_advance_window(
            algo_state_mgr=self._algo_state_mgr,
            current_step=self._run_step,
            total_steps=self.setting.num_split,
        ):
            try:
                self.load_next_window()
            except EOWSettingError:
                self._transition_state(
                    EvaluatorState.COMPLETED, allow_from=[EvaluatorState.STARTED, EvaluatorState.IN_PROGRESS]
                )
                raise RuntimeError("End of evaluation window reached")

        can_request, reason = self._algo_state_mgr.can_request_training_data(algo_id)
        if not can_request:
            raise PermissionError(f"Cannot request data: {reason}")
        # TODO handle case when algo is ready after submitting prediction, but current timestamp has not changed, meaning algo is requesting same data again
        self._algo_state_mgr.transition(
            algo_id=algo_id,
            new_state=AlgorithmStateEnum.RUNNING,
            data_segment=self._current_timestamp,
        )

        self._state = EvaluatorState.IN_PROGRESS
        # release data to the algorithm
        return self._training_data_cache

    def get_unlabeled_data(self, algo_id: UUID) -> PredictionMatrix:
        """Get unlabeled data for the algorithm.

        This method is called to get the unlabeled data for the algorithm. The
        unlabeled data is the data that the algorithm will predict. It will
        contain `(user_id, -1)` pairs, where the value -1 indicates that the
        item is to be predicted.
        """
        logger.debug(f"Getting unlabeled data for algorithm {algo_id}")
        can_submit, reason = self._algo_state_mgr.can_request_unlabeled_data(algo_id)
        if not can_submit:
            raise PermissionError(f"Cannot get unlabeled data: {reason}")
        return self._unlabeled_data_cache

    def submit_prediction(self, algo_id: UUID, X_pred: csr_matrix) -> None:
        """Submit the prediction of the algorithm.

        This method is called to submit the prediction of the algorithm.
        There are a few checks that are done before the prediction is
        evaluated by calling :meth:`_evaluate_algo_pred`.

        Once the prediction is evaluated, the method will update the state
        of the algorithm to PREDICTED.
        """
        logger.debug(f"Submitting prediction for algorithm {algo_id}")
        can_submit, reason = self._algo_state_mgr.can_submit_prediction(algo_id)
        if not can_submit:
            raise PermissionError(f"Cannot submit prediction: {reason}")

        self._evaluate_algo_pred(algorithm_id=algo_id, y_pred=X_pred)
        self._algo_state_mgr.transition(
            algo_id=algo_id,
            new_state=AlgorithmStateEnum.PREDICTED,
        )

    def _evaluate_algo_pred(self, algorithm_id: UUID, y_pred: csr_matrix) -> None:
        """Evaluate the prediction for algorithm."""
        y_true = self._ground_truth_data_cache.item_interaction_sequence_matrix

        if not self.ignore_unknown_item:
            y_pred = self._prediction_unknown_item_handler(y_true=y_true, y_pred=y_pred)
        algorithm_name = self._algo_state_mgr.get_algorithm_identifier(algo_id=algorithm_id)
        self._add_metric_results_for_prediction(
            ground_truth_data=self._ground_truth_data_cache,
            y_pred=y_pred,
            algorithm_name=algorithm_name,
        )

        logger.debug(f"Prediction evaluated for algorithm {algorithm_id} complete")
