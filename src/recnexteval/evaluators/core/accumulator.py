import logging
from collections import defaultdict
from typing import Optional

import pandas as pd

from ...metrics import Metric
from .constant import MetricLevelEnum


logger = logging.getLogger(__name__)


class MetricAccumulator:
    def __init__(self) -> None:
        self.acc: defaultdict[str, dict[str, Metric]] = defaultdict(dict)

    def __getitem__(self, key) -> dict[str, Metric]:
        return self.acc[key]

    def add(self, metric: Metric, algorithm_name: str) -> None:
        """Add a metric to the accumulator.

        Takes a Metric object and adds it under the algorithm name. If
        the specified metric already exists for the algorithm, it will be
        overwritten with the new metric.

        Args:
            metric: Metric to store.
            algorithm_name: Name of the algorithm.
        """
        if metric.identifier in self.acc[algorithm_name]:
            logger.warning(
                f"Metric {metric.identifier} already exists for algorithm {algorithm_name}. Overwriting..."
            )

        logger.debug(f"Metric {metric.identifier} created for algorithm {algorithm_name}")

        self.acc[algorithm_name][metric.identifier] = metric

    @property
    def user_level_metrics(self) -> defaultdict:
        results = defaultdict()
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)] = (
                    metric.micro_result
                )
        return results

    @property
    def window_level_metrics(self) -> defaultdict:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                score = metric.macro_result
                num_user = metric.num_users
                if score == 0 and num_user == 0:
                    logger.info(
                        f"Metric {metric.name} for algorithm {algo_name} "
                        f"at t={metric.timestamp_limit} has 0 score and 0 users. "
                        "The ground truth may be empty due to no interactions occurring in that window."
                    )
                elif score == 0 and num_user != 0:
                    logger.info(
                        f"Metric {metric.name} for algorithm {algo_name} "
                        f"at t={metric.timestamp_limit} has 0 score but there are interactions. "
                        f"{algo_name} did not have any correct predictions."
                    )
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["score"] = score
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["num_user"] = (
                    num_user
                )
        return results

    def df_user_level_metric(self) -> pd.DataFrame:
        """Get user-level metrics across all timestamps.

        Returns:
            DataFrame with user-level metric computations.
        """
        df = pd.DataFrame.from_dict(self.user_level_metrics, orient="index").explode(
            ["user_id", "score"]
        )
        df = df.rename_axis(["algorithm", "timestamp", "metric"])
        df.rename(columns={"score": "user_score"}, inplace=True)
        return df

    def df_window_level_metric(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.window_level_metrics, orient="index").explode(
            ["score", "num_user"]
        )
        df = df.rename_axis(["algorithm", "timestamp", "metric"])
        df.rename(columns={"score": "window_score"}, inplace=True)
        return df

    def df_macro_level_metric(self) -> pd.DataFrame:
        """Get macro-level metrics across all timestamps.

        Returns:
            DataFrame with macro-level metric computations.
        """
        df = pd.DataFrame.from_dict(self.window_level_metrics, orient="index").explode(
            ["score", "num_user"]
        )
        df = df.rename_axis(["algorithm", "timestamp", "metric"])
        result = df.groupby(["algorithm", "metric"]).mean()["score"].to_frame()
        result["num_window"] = df.groupby(["algorithm", "metric"]).count()["score"]
        result = result.rename(columns={"score": "macro_score"})
        return result

    def df_micro_level_metric(self) -> pd.DataFrame:
        """Get micro-level metrics across all timestamps.

        Returns:
            DataFrame with micro-level metric computations.
        """
        df = pd.DataFrame.from_dict(self.user_level_metrics, orient="index").explode(
            ["user_id", "score"]
        )
        df = df.rename_axis(["algorithm", "timestamp", "metric"])
        result = df.groupby(["algorithm", "metric"])["score"].mean().to_frame()
        result["num_user"] = df.groupby(["algorithm", "metric"])["score"].count()
        result = result.rename(columns={"score": "micro_score"})
        return result

    def df_metric(
        self,
        filter_timestamp: Optional[int] = None,
        filter_algo: Optional[str] = None,
        level: MetricLevelEnum = MetricLevelEnum.MACRO,
    ) -> pd.DataFrame:
        """Get DataFrame representation of metrics.

        Returns a DataFrame representation of the metrics. The DataFrame can be
        filtered based on algorithm name and timestamp.

        Args:
            filter_timestamp: Timestamp value to filter on. Defaults to None.
            filter_algo: Algorithm name to filter on. Defaults to None.
            level: Level of the metric to compute. Defaults to MetricLevelEnum.MACRO.

        Returns:
            DataFrame representation of the metrics.
        """
        if level == MetricLevelEnum.MACRO:
            df = self.df_macro_level_metric()
        elif level == MetricLevelEnum.MICRO:
            df = self.df_micro_level_metric()
        elif level == MetricLevelEnum.WINDOW:
            df = self.df_window_level_metric()
        elif level == MetricLevelEnum.USER:
            df = self.df_user_level_metric()
        else:
            raise ValueError("Invalid level specified")

        if filter_algo:
            df = df.filter(like=filter_algo, axis=0)
        if filter_timestamp:
            df = df.filter(like=f"t={filter_timestamp}", axis=0)
        return df
