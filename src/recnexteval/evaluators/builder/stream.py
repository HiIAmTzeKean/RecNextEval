import logging
from dataclasses import dataclass

from ..stream.evaluator_stream import EvaluatorStreamer
from .base import Builder


logger = logging.getLogger(__name__)


@dataclass
class EvaluatorStreamerBuilder(Builder):
    """Builder to facilitate construction of evaluator.

    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible
    errors when the evaluator is executed.
    """

    ignore_unknown_user: bool = True
    """Ignore unknown user in the evaluation"""
    ignore_unknown_item: bool = False
    """Ignore unknown item in the evaluation"""

    def build(self) -> EvaluatorStreamer:
        """Build Evaluator object.

        Raises:
            RuntimeError: If no metrics, algorithms or settings are specified.

        Returns:
            EvaluatorStreamer: The built evaluator object.
        """
        self._check_ready()
        return EvaluatorStreamer(
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            metric_k=self.metric_k,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
            seed=self.seed,
        )
