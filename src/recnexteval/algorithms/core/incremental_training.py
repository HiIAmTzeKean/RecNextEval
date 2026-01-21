from typing import Protocol, TypeVar

from scipy.sparse import csr_matrix, hstack, vstack


class HasX(Protocol):
    X_: csr_matrix | None


T = TypeVar("T", bound=HasX)


class IncrementalTrainingMixin:
    """Mixin providing a reusable method to append historical CSR training data.

    Expects the concrete class to have an attribute `X_` (csr_matrix | None).
    """

    def _append_training_data(self: T, X: csr_matrix) -> None:
        """Append a new interaction matrix to the historical data.

        Pads matrices to the same shape and sums them in-place into `self.X_`.
        """
        if getattr(self, "X_", None) is None:
            raise ValueError("No existing training data to append to.")

        X_prev: csr_matrix = self.X_.copy()
        new_num_rows = max(X_prev.shape[0], X.shape[0])
        new_num_cols = max(X_prev.shape[1], X.shape[1])

        # Pad the previous matrix
        if X_prev.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X_prev.shape[0], X_prev.shape[1]))
            X_prev = vstack([X_prev, row_padding])
        if X_prev.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X_prev.shape[0], new_num_cols - X_prev.shape[1]))
            X_prev = hstack([X_prev, col_padding])

        # Pad the current matrix
        if X.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X.shape[0], X.shape[1]))
            X = vstack([X, row_padding])
        if X.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X.shape[0], new_num_cols - X.shape[1]))
            X = hstack([X, col_padding])

        # Merge data
        self.X_ = X_prev + X
