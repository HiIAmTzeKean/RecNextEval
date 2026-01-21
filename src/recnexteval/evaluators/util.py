import logging
from dataclasses import dataclass, field
from enum import StrEnum

from ..matrix import InteractionMatrix


logger = logging.getLogger(__name__)


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


@dataclass
class UserItemBaseStatus:
    """Unknown and known user/item base.

    This class is used to store the status of the user and item base. The class
    stores the known and unknown user and item set. The class also provides
    methods to update the known and unknown user and item set.
    """

    unknown_user: set[int] = field(default_factory=set)
    known_user: set[int] = field(default_factory=set)
    unknown_item: set[int] = field(default_factory=set)
    known_item: set[int] = field(default_factory=set)

    @property
    def known_shape(self) -> tuple[int, int]:
        """Known number of user id and item id.

        id are zero-indexed and the shape returns the max id + 1.

        Note:
            `max` is used over `len` as there may be gaps in the id sequence
            and we are only concerned with the shape of the
            user-item interaction matrix.

        Returns:
            Tuple of (|user|, |item|).
        """
        return (max(self.known_user) + 1, max(self.known_item) + 1)

    @property
    def global_shape(self) -> tuple[int, int]:
        """Global number of user id and item id.

        This is the shape of the user-item interaction matrix considering all
        the users and items that has been possibly exposed. The global shape
        considers the fact that an unknown user/item can be exposed during the
        prediction stage when an unknown user/item id is requested for prediction
        on the algorithm.

        Returns:
            Tuple of (|user|, |item|).
        """
        return (
            max(max(self.known_user), max(self.unknown_user)) + 1,
            max(max(self.known_item), max(self.unknown_item)) + 1,
        )

    def update_known_user_item_base(self, data: InteractionMatrix) -> None:
        """Updates the known user and item set with the data."""
        self.known_item.update(data.item_ids)
        self.known_user.update(data.user_ids)

    def update_unknown_user_item_base(self, data: InteractionMatrix) -> None:
        """Updates the unknown user and item set with the data. """
        self.unknown_user = data.user_ids - self.known_user
        self.unknown_item = data.item_ids - self.known_item

    def reset_unknown_user_item_base(self) -> None:
        """Clears the unknown user and item set."""
        self.unknown_user.clear()
        self.unknown_item.clear()
