from typing import NamedTuple

from recnexteval.matrix import InteractionMatrix


class SplitResult(NamedTuple):
    """Named tuple for split data results."""
    unlabeled: InteractionMatrix
    ground_truth: InteractionMatrix
    incremental: InteractionMatrix | None
    t_window: int | None
