import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ...utils import add_rows_to_csr_matrix


logger = logging.getLogger(__name__)


class PopularityPaddingMixin:
    """Mixin class to add popularity-based padding to prediction methods."""

    def __init__(self, pad_with_popularity: bool = False) -> None:
        super().__init__()
        self.pad_with_popularity = pad_with_popularity

    def get_popularity_scores(self, X: csr_matrix) -> np.ndarray:
        """Compute a popularity-based scoring vector for items.

        This method calculates normalized interaction counts for each item,
        selects the top-K most popular items, and returns a vector where
        only those top-K items have their normalized scores (others are 0).
        This is used to pad predictions for unseen users with popular items.

        :param X: The interaction matrix (user-item) to compute popularity from.
        :type X: csr_matrix
        :return: A 1D array of shape (num_items,) with popularity scores for top-K items.
        :rtype: np.ndarray
        """
        interaction_counts = X.sum(axis=0).A[0]
        normalized_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if hasattr(self, "K"):
            k_value = self.K
        else:
            k_value = 100
        if num_items < k_value:
            logger.warning("K is larger than the number of items.")

        effective_k = min(k_value, num_items)
        # Get indices of top-K items by popularity
        top_k_indices = np.argpartition(normalized_scores, -effective_k)[-effective_k:]
        popularity_vector = np.zeros(num_items)
        popularity_vector[top_k_indices] = normalized_scores[top_k_indices]

        return popularity_vector

    def _pad_unknown_uid_with_popularity_strategy(
        self,
        X_pred: csr_matrix,
        intended_shape: tuple,
        predict_ui_df: pd.DataFrame,
    ) -> csr_matrix:
        """Pad the predictions with popular items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple
        :param predict_ui_df: DataFrame containing the user IDs to predict for
        :type predict_ui_df: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        # pad users with popular items
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with popular items")
        popular_items = self.get_popularity_scores(X_pred)

        to_predict = predict_ui_df.value_counts("uid")
        # Filter for users not in training data
        filtered = to_predict[to_predict.index >= known_user_id]
        for user_id in filtered.index:
            if user_id >= known_user_id:
                X_pred[user_id, :] = popular_items
        return X_pred