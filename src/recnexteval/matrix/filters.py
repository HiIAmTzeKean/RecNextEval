import logging
import operator
from typing import TYPE_CHECKING, Literal, TypeVar, overload


if TYPE_CHECKING:
    from .interaction_matrix import InteractionMatrix  # noqa: F401

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="InteractionMatrix")


class SelectionIDMixin:
    @overload
    def users_in(self: T, U: set[int]) -> T: ...
    @overload
    def users_in(self: T, U: set[int], inplace: Literal[False]) -> T: ...
    @overload
    def users_in(self: T, U: set[int], inplace: Literal[True]) -> None: ...
    def users_in(self: T, U: set[int], inplace: bool = False) -> None | T:
        logger.debug("Performing users_in comparison")
        mask = self._df[self.USER_IX].isin(U)
        return self._apply_mask(mask, inplace=inplace)

    @overload
    def users_not_in(self: T, U: set[int]) -> T: ...
    @overload
    def users_not_in(self: T, U: set[int], inplace: Literal[False]) -> T: ...
    @overload
    def users_not_in(self: T, U: set[int], inplace: Literal[True]) -> None: ...
    def users_not_in(self: T, U: set[int], inplace: bool = False) -> None | T:
        logger.debug("Performing users_not_in comparison")
        mask = ~self._df[self.USER_IX].isin(U)
        return self._apply_mask(mask, inplace=inplace)

    @overload
    def items_in(self: T, id_set: set[int]) -> T: ...
    @overload
    def items_in(self: T, id_set: set[int], inplace: Literal[False]) -> T: ...
    @overload
    def items_in(self: T, id_set: set[int], inplace: Literal[True]) -> None: ...
    def items_in(self: T, id_set: set[int], inplace: bool = False) -> None | T:
        logger.debug("Performing items_in comparison")
        mask = self._df[self.ITEM_IX].isin(id_set)
        return self._apply_mask(mask, inplace=inplace)

    @overload
    def items_not_in(self: T, id_set: set[int]) -> T: ...
    @overload
    def items_not_in(self: T, id_set: set[int], inplace: Literal[False]) -> T: ...
    @overload
    def items_not_in(self: T, id_set: set[int], inplace: Literal[True]) -> None: ...
    def items_not_in(self: T, id_set: set[int], inplace: bool = False) -> None | T:
        logger.debug("Performing items_not_in comparison")
        mask = ~self._df[self.ITEM_IX].isin(id_set)
        return self._apply_mask(mask, inplace=inplace)


class SelectionTimestampMixin:
    @overload
    def timestamps_gt(self: T, timestamp: float) -> T: ...
    @overload
    def timestamps_gt(self: T, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_gt(self: T, timestamp: float, inplace: bool = False) -> None | T:
        """Select interactions after a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.gt, timestamp, inplace)

    @overload
    def timestamps_gte(self: T, timestamp: float) -> T: ...
    @overload
    def timestamps_gte(self: T, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_gte(self: T, timestamp: float, inplace: bool = False) -> None | T:
        """Select interactions after and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.ge, timestamp, inplace)

    @overload
    def timestamps_lt(self: T, timestamp: float) -> T: ...
    @overload
    def timestamps_lt(self: T, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_lt(self: T, timestamp: float, inplace: bool = False) -> None | T:
        """Select interactions up to a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.lt, timestamp, inplace)

    @overload
    def timestamps_lte(self: T, timestamp: float) -> T: ...
    @overload
    def timestamps_lte(self: T, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_lte(self: T, timestamp: float, inplace: bool = False) -> None | T:
        """Select interactions up to and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.le, timestamp, inplace)
