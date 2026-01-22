from enum import StrEnum


class AlgorithmStateEnum(StrEnum):
    """Enum for the state of the algorithm.

    Used to keep track of the state of the algorithm during the streaming
    process in the `EvaluatorStreamer`.
    """

    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    PREDICTED = "PREDICTED"
    COMPLETED = "COMPLETED"


class MetricLevelEnum(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    WINDOW = "window"
    USER = "user"

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check valid value for MetricLevelEnum.

        Args:
            value: String value input.

        Returns:
            Whether the value is valid.
        """
        return value in MetricLevelEnum
