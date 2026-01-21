from enum import StrEnum


class ItemUserBasedEnum(StrEnum):
    """Enum class for item and user based properties.

    Enum class to indicate if the function or logic is based on item or user.
    """

    ITEM = "item"
    """Property based on item"""
    USER = "user"
    """Property based on user"""

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check valid value for ItemUserBasedEnum

        :param value: String value input
        :type value: str
        """
        return value in ItemUserBasedEnum
