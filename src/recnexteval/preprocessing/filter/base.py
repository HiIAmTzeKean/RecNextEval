"""Data filtering module.

This module provides abstract base class and filter implementations for
removing interactions from a DataFrame based on various criteria.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Filter(ABC):
    """Abstract base class for filter implementations.

    A filter must implement an `apply` method that takes a pandas DataFrame
    as input and returns a processed pandas DataFrame as output.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to the DataFrame.

        Args:
            df: DataFrame to filter.

        Returns:
            Filtered DataFrame.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        attrs = self.__dict__
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"
