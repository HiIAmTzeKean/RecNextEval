import pandas as pd

from .base import Filter


class MinItemsPerUser(Filter):
    """Filter requiring users to have minimum interaction count.

    Removes users who have interacted with fewer than the specified minimum
    number of items. Adapted from RecPack.

    Args:
        min_items_per_user: Minimum number of items a user must interact with.
        item_ix: Column name containing item identifiers.
        user_ix: Column name containing user identifiers.
        count_duplicates: Whether to count multiple interactions with the same
            item. Defaults to True.

    Example:
        Original interactions:
        ```
        user | item
        1 | a
        1 | b
        1 | c
        2 | a
        2 | b
        2 | d
        3 | a
        3 | b
        3 | d
        ```

        After `MinItemsPerUser(3)`:
        ```
        user | item
        1 | a
        1 | b
        2 | a
        2 | b
        3 | a
        3 | b
        ```

        Users 1 and 2 are removed (have fewer than 3 items).
    """

    def __init__(
        self,
        min_items_per_user: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = True,
    ) -> None:
        self.min_items_per_user = min_items_per_user
        self.count_duplicates = count_duplicates
        self.item_ix = item_ix
        self.user_ix = user_ix

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum items per user filter.

        Args:
            df: DataFrame to filter.

        Returns:
            DataFrame containing only users with sufficient interactions.
        """
        uids = (
            df[self.user_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.user_ix]
        )

        cnt_items_per_user = uids.value_counts()
        users_of_interest = list(cnt_items_per_user[cnt_items_per_user >= self.min_items_per_user].index)
        return df[df[self.user_ix].isin(users_of_interest)].copy()
