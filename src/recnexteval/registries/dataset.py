import importlib

from ..datasets import Dataset
from .base import Registry


class DatasetRegistry(Registry[Dataset]):
    """Registry for easy retrieval of dataset types by name.

    The registry comes preregistered with all the recnexteval datasets.
    """

    def __init__(self) -> None:
        module = importlib.import_module('recnexteval.datasets')
        super().__init__(module)


DATASET_REGISTRY = DatasetRegistry()
"""Registry for datasets.

Contains the recnexteval metrics by default,
and allows registration of new metrics via the `register` function.

Example:
    ```python
    from recnexteval.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from recnexteval.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)
    ```
"""
