from scipy.sparse import csr_matrix, hstack, vstack


class IncrementalTrainingMixin:
    """Mixin providing a reusable method to append historical CSR training data.

    Expects the concrete class to have an attribute `X_` (csr_matrix | None).
    """
    X_: csr_matrix | None

    @staticmethod
    def _pad_sparse_matrix(X: csr_matrix, target_shape: tuple[int, int]) -> csr_matrix:
        if X.shape == target_shape:
            return X
        if X.shape[0] < target_shape[0]:
            row_pad = csr_matrix((target_shape[0] - X.shape[0], X.shape[1]))
            X = vstack([X, row_pad])
        if X.shape[1] < target_shape[1]:
            col_pad = csr_matrix((X.shape[0], target_shape[1] - X.shape[1]))
            X = hstack([X, col_pad])
        return X

    def _append_training_data(self, X: csr_matrix) -> None:
        """Append a new interaction matrix to the historical data.

        Pads matrices to the same shape and sums them in-place into `self.X_`.
        """
        if self.X_ is None:
            raise ValueError("X_ must be initialized first")

        target_shape = (max(self.X_.shape[0], X.shape[0]), max(self.X_.shape[1], X.shape[1]))
        X_prev = self._pad_sparse_matrix(self.X_.copy(), target_shape)
        X = self._pad_sparse_matrix(X, target_shape)
        self.X_ = X_prev + X
