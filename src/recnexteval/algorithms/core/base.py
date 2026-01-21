import logging
import time
from abc import abstractmethod
from inspect import Parameter, signature
from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from recnexteval.matrix import InteractionMatrix, ItemUserBasedEnum, PredictionMatrix, to_csr_matrix
from ...models import BaseModel, ParamMixin
from ...utils import add_columns_to_csr_matrix, add_rows_to_csr_matrix


logger = logging.getLogger(__name__)


class Algorithm(BaseEstimator, BaseModel, ParamMixin):
    """Base class for all recnexteval algorithm implementations."""

    ITEM_USER_BASED: ItemUserBasedEnum

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "seed"):
            self.seed = 42
        self.rand_gen = np.random.default_rng(seed=self.seed)

    @property
    def description(self) -> str:
        """Description of the algorithm.

        :return: Description of the algorithm
        :rtype: str
        """
        return self.__doc__ or "No description provided."

    @property
    def identifier(self) -> str:
        """Identifier of the object.

        Identifier is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: `Algorithm(param_1=value)`

        :return: Identifier of the object
        :rtype: str
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    @classmethod
    def get_default_params(cls) -> dict:
        """Get default parameters without instantiation.

        Uses inspect.signature to extract __init__ parameters and their
        default values without instantiating the class.

        Returns:
            Dictionary of parameter names to default values.
            Parameters without defaults map to None.
        """
        try:
            sig = signature(cls.__init__)
        except (ValueError, TypeError):
            # Fallback for built-in types or special cases
            return {}

        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                # Skip *args, **kwargs
                continue

            # Extract the default value
            if param.default is not Parameter.empty:
                params[param_name] = param.default
            else:
                params[param_name] = None

        return params

    def __str__(self) -> str:
        return self.name

    def set_params(self, **params) -> Self:
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        return super().set_params(**params)

    @abstractmethod
    def _fit(self, X: csr_matrix) -> Self:
        """Stub implementation for fitting an algorithm.

        Will be called by the `fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    @abstractmethod
    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        """Stub for predicting scores to users

        Will be called by the `predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: PredictionMatrix
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self) -> None:
        """Helper function to check if model was correctly fitted

        Uses the sklearn check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

    def fit(self, X: InteractionMatrix) -> Self:
        """Fit the model to the input interaction matrix.

        The input data is transformed to the expected type using
        :meth:`_transform_fit_input`. The fitting is done using the
        :meth:`_fit` method. Finally the method checks that the fitting
        was successful using :meth:`_check_fit_complete`.

        :param X: The interactions to fit the model on.
        :type X: InteractionMatrix
        :return: Fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        X_transformed = to_csr_matrix(X, binary=True)
        self._fit(X_transformed)

        self._check_fit_complete()
        end = time.time()
        logger.debug(f"Fitting {self.name} complete - Took {end - start:.3}s")
        return self

    def _pad_unknown_iid_with_none_strategy(
        self,
        y_pred: csr_matrix,
        current_shape: tuple[int, int],
        intended_shape: tuple[int, int],
    ) -> csr_matrix:
        """Pad the predictions with empty fields for unknown items.

        This is to ensure that when we compute the performance of the prediction, we are
        comparing the prediction against the ground truth for the same set of items.
        """
        if y_pred.shape == intended_shape:
            return y_pred

        known_user_id, known_item_id = current_shape
        logger.debug(f"Padding item ID in range({known_item_id}, {intended_shape[1]}) with empty fields")
        y_pred = add_columns_to_csr_matrix(y_pred, intended_shape[1] - known_item_id)
        logger.debug(f"Padding by {self.name} completed")
        return y_pred

    # TODO change X_pred to y_pred for consistency
    def _pad_unknown_uid_with_random_strategy(
        self,
        X_pred: csr_matrix,
        current_shape: tuple[int, int],
        intended_shape: tuple[int, int],
        predict_ui_df: pd.DataFrame,
        k: int = 10,
    ) -> csr_matrix:
        """Pad the predictions with random items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple[int, int]
        :param predict_ui_df: DataFrame containing the user IDs to predict for
        :type predict_ui_df: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = current_shape
        # +1 to include the last user id
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        # pad users with random items
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with random items")
        to_predict = pd.Series(predict_ui_df.uid.unique())
        # Filter for users not in training data
        filtered = to_predict[to_predict >= known_user_id]
        filtered = filtered.sort_values(ignore_index=True)
        if not filtered.empty:
            row = filtered.values.repeat(k)
            total_pad = len(row)
            col = self.rand_gen.integers(0, known_item_id, total_pad)
            pad = csr_matrix((np.ones(total_pad), (row, col)), shape=intended_shape)
        else:
            pad = csr_matrix(intended_shape)
        X_pred += pad
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def predict(self, X: PredictionMatrix) -> csr_matrix:
        """Predicts scores, given the interactions in X

        The input data is transformed to the expected type using
        :meth:`_transform_predict_input`. The predictions are made
        using the :meth:`_predict` method. Finally the predictions
        are then padded with random items for users that are not in the
        training data.

        :param X: interactions to predict from.
        :type X: InteractionMatrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        self._check_fit_complete()
        X_pred = self._predict(X)
        return X_pred
