from enum import Enum


class EvaluatorState(Enum):
    """Evaluator lifecycle states"""

    INITIALIZED = "initialized"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
