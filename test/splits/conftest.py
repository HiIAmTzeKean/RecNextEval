from typing import Any

import pandas as pd
import pytest

from recnexteval.matrix import InteractionMatrix


@pytest.fixture
def matrix(test_dataframe: pd.DataFrame, session_vars: dict) -> InteractionMatrix:
    return InteractionMatrix(
        df=test_dataframe,
        item_ix=session_vars["ITEM_IX"],
        user_ix=session_vars["USER_IX"],
        timestamp_ix=session_vars["TIMESTAMP_IX"],
    )


@pytest.fixture
def empty_matrix(session_vars: dict) -> InteractionMatrix:
    """Fixture for an empty interaction matrix."""
    df = pd.DataFrame(
        columns=[
            session_vars["TIMESTAMP_IX"],
            session_vars["USER_IX"],
            session_vars["ITEM_IX"],
            "rating",
        ]
    )
    return InteractionMatrix(
        df=df,
        item_ix=session_vars["ITEM_IX"],
        user_ix=session_vars["USER_IX"],
        timestamp_ix=session_vars["TIMESTAMP_IX"],
    )


@pytest.fixture
def matrix_no_timestamps(test_dataframe: pd.DataFrame, session_vars: dict[str, Any]) -> InteractionMatrix:
    """Fixture for matrix without timestamps."""
    df = test_dataframe.drop(columns=[session_vars["TIMESTAMP_IX"]])
    # Create matrix with skip_df_processing to avoid requiring timestamp column
    im = InteractionMatrix(
        df=df,
        item_ix=session_vars["ITEM_IX"],
        user_ix=session_vars["USER_IX"],
        timestamp_ix="dummy",
        skip_df_processing=True,
    )
    # Remove the timestamp column to simulate no timestamps
    im._df = im._df.drop(columns=[InteractionMatrix.TIMESTAMP_IX], errors="ignore")
    return im
