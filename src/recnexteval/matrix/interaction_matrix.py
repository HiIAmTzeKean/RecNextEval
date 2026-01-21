import logging
from copy import deepcopy
from typing import Literal, Self, overload

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .enums import ItemUserBasedEnum
from .exception import TimestampAttributeMissingError
from .filters import SelectionIDMixin, SelectionTimestampMixin


logger = logging.getLogger(__name__)


class InteractionMatrix(SelectionIDMixin, SelectionTimestampMixin):
    """Matrix of interaction data between users and items.

    It provides a number of properties and methods for easy manipulation of this interaction data.

    .. attention::

        - The InteractionMatrix does not assume binary user-item pairs.
          If a user interacts with an item more than once, there will be two
          entries for this user-item pair.

        - We assume that the user and item IDs are integers starting from 0. IDs
          that are indicated by "-1" are reserved to label the user or item to
          be predicted. This assumption is crucial as it will be used during the
          split scheme and evaluation of the RS since it will affect the 2D shape
          of the CSR matrix
    """

    ITEM_IX = "iid"
    USER_IX = "uid"
    TIMESTAMP_IX = "ts"
    INTERACTION_IX = "interactionid"
    MASKED_LABEL = -1

    def __init__(
        self,
        df: pd.DataFrame,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str,
        shape: None | tuple[int, int] = None,
        skip_df_processing: bool = False,
    ) -> None:
        self.user_item_shape: tuple[int, int]
        """The shape of the interaction matrix, i.e. `|user| x |item|`."""
        if shape:
            self.user_item_shape = shape

        if skip_df_processing:
            self._df = df
            return

        col_mapper = {
            item_ix: self.ITEM_IX,
            user_ix: self.USER_IX,
            timestamp_ix: self.TIMESTAMP_IX,
        }
        df = df.rename(columns=col_mapper)
        required_columns = [
            self.USER_IX,
            self.ITEM_IX,
            self.TIMESTAMP_IX,
        ]
        extra_columns = [col for col in df.columns if col not in required_columns]
        df = df[required_columns + extra_columns].copy()
        # TODO refactor this statement below
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": InteractionMatrix.INTERACTION_IX})

        self._df = df

    def copy(self) -> Self:
        """Create a deep copy of this InteractionMatrix."""
        return deepcopy(self)

    def copy_df(self, reset_index: bool = False) -> "pd.DataFrame":
        """Create a deep copy of the dataframe."""
        if reset_index:
            return deepcopy(self._df.reset_index(drop=True))
        return deepcopy(self._df)

    def concat(self, im: "InteractionMatrix | pd.DataFrame") -> Self:
        """Concatenate this InteractionMatrix with another.

        Note:
            This is a inplace operation. and will modify the current object.
        """
        if isinstance(im, pd.DataFrame):
            self._df = pd.concat([self._df, im])
        else:
            self._df = pd.concat([self._df, im._df])

        return self

    # TODO this should be shifted to prediction matrix
    def union(self, im: "InteractionMatrix") -> Self:
        """Combine events from this InteractionMatrix with another."""
        return self + im

    def difference(self, im: "InteractionMatrix") -> Self:
        """Difference between this InteractionMatrix and another."""
        return self - im

    @property
    def values(self) -> csr_matrix:
        """All user-item interactions as a sparse matrix of size (|users|, |items|).

        The shape of the matrix is determined by the `user_item_shape` attribute. Each row represents
        a user and each column represents an item. The index of the rows and columns correspond to the user
        and item IDs respectively. An entry in the matrix is 1 if there is an interaction.
        """
        # TODO issue with -1 labeling in the interaction matrix should i create prediction matrix
        if not hasattr(self, "user_item_shape"):
            raise AttributeError(
                "InteractionMatrix has no `user_item_shape` attribute. Please call mask_shape() first."
            )

        values = np.ones(self._df.shape[0])
        indices = self._df[[self.USER_IX, self.ITEM_IX]].values
        indices = (indices[:, 0], indices[:, 1])

        matrix = csr_matrix((values, indices), shape=self.user_item_shape, dtype=np.int32)
        return matrix

    @property
    def indices(self) -> tuple[list[int], list[int]]:
        """Returns a tuple of lists of user IDs and item IDs corresponding to interactions.

        :return: tuple of lists of user IDs and item IDs that correspond to at least one interaction.
        :rtype: tuple[list[int], list[int]]
        """
        return self.values.nonzero()

    def nonzero(self) -> tuple[list[int], list[int]]:
        return self.values.nonzero()

    @overload
    def _apply_mask(self, mask: pd.Series) -> Self: ...
    @overload
    def _apply_mask(self, mask: pd.Series, inplace: Literal[True]) -> None: ...
    @overload
    def _apply_mask(self, mask: pd.Series, inplace: Literal[False]) -> Self: ...
    def _apply_mask(self, mask: pd.Series, inplace: bool = False) -> None | Self:
        interaction_m = self if inplace else self.copy()
        interaction_m._df = interaction_m._df[mask]
        return None if inplace else interaction_m

    # Timestamp selection helpers moved to SelectionTimestampMixin (src/recnexteval/matrix/filters.py)

    def __add__(self, im: "InteractionMatrix") -> Self:
        """Combine events from this InteractionMatrix with another.

        :param im: InteractionMatrix to union with.
        :type im: InteractionMatrix
        :return: Union of interactions in this InteractionMatrix and the other.
        :rtype: InteractionMatrix
        """
        df = pd.concat([self._df, im._df], copy=False)

        shape = None
        if hasattr(self, "user_item_shape") and hasattr(im, "user_item_shape"):
            shape = (
                max(self.user_item_shape[0], im.user_item_shape[0]),
                max(self.user_item_shape[1], im.user_item_shape[1]),
            )
            self.user_item_shape = shape

        return type(self)(
            df,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
            shape,
            True,
        )

    def __sub__(self, im: "InteractionMatrix") -> Self:
        full_data = pd.MultiIndex.from_frame(self._df)
        data_part_2 = pd.MultiIndex.from_frame(im._df)
        data_part_1 = full_data.difference(data_part_2).to_frame().reset_index(drop=True)

        shape = None
        if hasattr(self, "user_item_shape") and hasattr(im, "user_item_shape"):
            shape = (
                max(self.user_item_shape[0], im.user_item_shape[0]),
                max(self.user_item_shape[1], im.user_item_shape[1]),
            )

        return type(self)(
            data_part_1,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
            shape,
            True,
        )

    def __repr__(self) -> str:
        return repr(self._df)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InteractionMatrix):
            logger.debug(f"Comparing {type(value)} with InteractionMatrix is not supported")
            return False
        return self._df.equals(value._df)

    def __len__(self) -> int:
        """Return the number of interactions in the matrix.

        This is distinct from the shape of the matrix, which is the number of
        users and items that has been released to the model. The length of the
        matrix is the number of interactions present in the matrix resulting
        from filter operations.
        """
        return len(self._df)

    def _get_last_n_interactions(
        self,
        by: ItemUserBasedEnum,
        n_seq_data: int,
        t_upper: None | int = None,
        id_in: None | set[int] = None,
        inplace=False,
    ) -> Self:
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        if t_upper is None:
            t_upper = self.max_timestamp + 1  # to include the last timestamp

        interaction_m = self if inplace else self.copy()

        mask = interaction_m._df[InteractionMatrix.TIMESTAMP_IX] < t_upper
        if id_in and by == ItemUserBasedEnum.USER:
            mask = mask & interaction_m._df[InteractionMatrix.USER_IX].isin(id_in)
        elif id_in and by == ItemUserBasedEnum.ITEM:
            mask = mask & interaction_m._df[InteractionMatrix.ITEM_IX].isin(id_in)

        if by == ItemUserBasedEnum.USER:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.USER_IX).tail(n_seq_data)
        else:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.ITEM_IX).tail(n_seq_data)
        interaction_m._df = c_df

        return interaction_m

    def _get_first_n_interactions(
        self, by: ItemUserBasedEnum, n_seq_data: int, t_lower: None | int = None, inplace=False
    ) -> Self:
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        if t_lower is None:
            t_lower = self.min_timestamp

        interaction_m = self if inplace else self.copy()

        mask = interaction_m._df[InteractionMatrix.TIMESTAMP_IX] >= t_lower
        if by == ItemUserBasedEnum.USER:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.USER_IX).head(n_seq_data)
        else:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.ITEM_IX).head(n_seq_data)
        interaction_m._df = c_df
        return interaction_m

    @property
    def item_interaction_sequence_matrix(self) -> csr_matrix:
        """Converts the interaction data into an item interaction sequence matrix.

        Dataframe values are converted into such that the row sequence is maintained and
        the item that interacted with will have the column at item_id marked with 1.
        """
        values = np.ones(self._df.shape[0])
        indices = (np.arange(self._df.shape[0]), self._df[self.ITEM_IX].to_numpy())
        shape = (self._df.shape[0], self.user_item_shape[1])

        sparse_matrix = csr_matrix((values, indices), shape=shape, dtype=values.dtype)
        return sparse_matrix

    @property
    def user_id_sequence_array(self) -> np.ndarray:
        """Array of user IDs in the order of interactions.

        :return: Numpy array of user IDs.
        :rtype: np.ndarray
        """
        return self._df[InteractionMatrix.USER_IX].to_numpy()

    @property
    def user_ids(self) -> set[int]:
        """The set of all user ID in matrix"""
        return set(self._df[InteractionMatrix.USER_IX].dropna().unique())

    @property
    def item_ids(self) -> set[int]:
        """The set of all item ID in matrix"""
        return set(self._df[InteractionMatrix.ITEM_IX].dropna().unique())

    @property
    def num_interactions(self) -> int:
        """The total number of interactions.

        :return: Total interaction count.
        :rtype: int
        """
        return len(self._df)

    @property
    def has_timestamps(self) -> bool:
        """Boolean indicating whether instance has timestamp information.

        :return: True if timestamps information is available, False otherwise.
        :rtype: bool
        """
        return self.TIMESTAMP_IX in self._df

    @property
    def min_timestamp(self) -> int:
        """The earliest timestamp in the interaction

        :return: The earliest timestamp.
        :rtype: int
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        return self._df[self.TIMESTAMP_IX].min()

    @property
    def max_timestamp(self) -> int:
        """The latest timestamp in the interaction

        :return: The latest timestamp.
        :rtype: int
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        return self._df[self.TIMESTAMP_IX].max()

    @property
    def global_num_user(self) -> int:
        return max(int(self._df[InteractionMatrix.USER_IX].max()) + 1, self.user_item_shape[0] + 1)

    @property
    def global_num_item(self) -> int:
        return max(int(self._df[InteractionMatrix.ITEM_IX].max()) + 1, self.user_item_shape[1] + 1)

    @property
    def known_num_user(self) -> int:
        """The highest known number of users

        Note that we add 1 to the max known user ID to get the number of users,
        since user IDs are zero-indexed.
        """
        max_val = self._df[(self._df != -1).all(axis=1)][InteractionMatrix.USER_IX].max()
        if pd.isna(max_val):
            return self.user_item_shape[0]
        return min(int(max_val) + 1, self.user_item_shape[0] + 1)

    @property
    def known_num_item(self) -> int:
        """The highest known user ID in the interaction matrix."""
        max_val = self._df[(self._df != -1).all(axis=1)][InteractionMatrix.ITEM_IX].max()
        if pd.isna(max_val):
            return self.user_item_shape[1]
        return min(int(max_val) + 1, self.user_item_shape[1] + 1)

    # TODO deprecate these two properties
    @property
    def max_user_id(self) -> int:
        """The highest known user ID in the interaction matrix.

        :return: The highest user ID.
        :rtype: int
        """
        max_val = self._df[self._df != -1][InteractionMatrix.USER_IX].max()
        if np.isnan(max_val):
            return -1
        return max_val

    @property
    def max_item_id(self) -> int:
        """The highest known item ID in the interaction matrix.

        In the case of an empty matrix, the highest item ID is -1. This is
        consistent with the the definition that -1 denotes the item that is
        unknown. It would be incorrect to use any other value, since 0 is a
        valid item ID.

        :return: The highest item ID.
        :rtype: int
        """
        max_val = self._df[self._df != -1][InteractionMatrix.ITEM_IX].max()
        if np.isnan(max_val):
            return -1
        return max_val

    @property
    def timestamps(self) -> pd.Series:
        """Timestamps of interactions as a pandas Series, indexed by user ID and item ID.

        :raises TimestampAttributeMissingError: If timestamp column is missing.
        :return: Interactions with composite index on (user ID, item ID)
        :rtype: pd.Series
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        index = pd.MultiIndex.from_frame(self._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]])
        return pd.DataFrame(self._df[[InteractionMatrix.TIMESTAMP_IX]]).set_index(index)[InteractionMatrix.TIMESTAMP_IX]

    @property
    def latest_interaction_timestamps_matrix(self) -> csr_matrix:
        """A csr matrix containing the last interaction timestamp for each user, item pair.

        We only account for the last interacted timestamp making the dataset non-deduplicated.
        """
        timestamps = self.timestamps.groupby(["uid", "iid"]).max().reset_index()
        timestamp_mat = csr_matrix(
            (timestamps.ts.values, (timestamps.uid.values, timestamps.iid.values)),
            shape=self.user_item_shape,
        )

        return timestamp_mat
