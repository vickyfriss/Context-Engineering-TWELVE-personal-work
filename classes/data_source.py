"""Data source classes used across multiple Streamlit pages.

This module supports:
- Player-level stats
- Country-level stats
- Personality/person-level stats
- Team-level stats for the team build-up analyst

Important:
- TeamStats automatically resolves the CSV file from the configured folder,
  so the exact file name does not matter as long as there is exactly one CSV
  file in that folder.
"""

from __future__ import annotations

import os
from math import floor, ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore

import classes.data_point as data_point


# Store current working directory in a reusable variable.
working_dir = Path(__file__).resolve().parents[1]


class Data:
    """Base class for all data sources."""

    data_point_class = None

    def __init__(self) -> None:
        """Load and process data during initialization."""
        self.df = self.get_processed_data()

    def get_raw_data(self) -> pd.DataFrame:
        """Load raw data.

        Returns:
            pd.DataFrame: Raw dataframe.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement get_raw_data().")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into the final dataframe.

        Args:
            df_raw (pd.DataFrame): Raw dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement process_data(df_raw).")

    def get_processed_data(self) -> pd.DataFrame:
        """Load raw data and return processed data.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        raw = self.get_raw_data()
        return self.process_data(raw)

    def select_and_filter(
        self,
        column_name: str,
        label: str,
        default_index: int = 0,
    ) -> None:
        """Filter the dataframe using a Streamlit selectbox.

        Args:
            column_name (str): Column used for filtering.
            label (str): Streamlit label for the selectbox.
            default_index (int): Default selected index.
        """
        df = self.df
        selected_id = st.selectbox(label, df[column_name].unique(), index=default_index)
        self.df = df[df[column_name] == selected_id]


class Stats(Data):
    """Base class for data sources that use metric-based statistics."""

    def __init__(self) -> None:
        """Initialize the stats object and metric containers."""
        self.df = self.get_processed_data()
        self.metrics: list[str] = []
        self.negative_metrics: list[str] = []

    def get_metric_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-scores for metric columns.

        If a metric is listed in `negative_metrics`, the sign is flipped
        so that higher normalized values always represent better outcomes.

        Args:
            df (pd.DataFrame): Dataframe with metric columns only.

        Returns:
            pd.DataFrame: Dataframe with '<metric>_Z' columns.
        """
        df_z = df.apply(zscore, nan_policy="omit")
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        for metric in set(self.negative_metrics).intersection(self.metrics):
            z_col = f"{metric}_Z"
            if z_col in df_z.columns:
                df_z[z_col] = df_z[z_col] * -1

        return df_z

    def get_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate descending ranks for metric columns.

        Args:
            df (pd.DataFrame): Dataframe with metric columns.

        Returns:
            pd.DataFrame: Dataframe with '<metric>_Ranks' columns.
        """
        df_ranks = df.rank(ascending=False)
        df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]
        return df_ranks

    def get_pct_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks for metric columns.

        Args:
            df (pd.DataFrame): Dataframe with metric columns.

        Returns:
            pd.DataFrame: Dataframe with '<metric>_Pct_Ranks' columns.
        """
        df_pct = df.rank(pct=True) * 100
        df_pct.columns = [f"{col}_Pct_Ranks" for col in df_pct.columns]
        return df_pct

    def calculate_statistics(
        self,
        metrics: list[str],
        negative_metrics: list[str] | None = None,
        include_pct_ranks: bool = False,
    ) -> None:
        """Calculate z-scores, ranks, and optional percentile ranks.

        Args:
            metrics (list[str]): Metric columns to include.
            negative_metrics (list[str] | None): Metrics where lower values
                are better. These will be sign-flipped in z-score space.
            include_pct_ranks (bool): If True, append percentile ranks too.

        Raises:
            ValueError: If one or more metrics are missing from the dataframe.
        """
        if negative_metrics is None:
            negative_metrics = []

        missing_metrics = [metric for metric in metrics if metric not in self.df.columns]
        if missing_metrics:
            raise ValueError(
                f"These metrics were not found in the dataframe: {missing_metrics}"
            )

        self.metrics = metrics
        self.negative_metrics = negative_metrics

        df = self.df.copy()
        df_metric_zscores = self.get_metric_zscores(df[metrics])

        df_z_for_ranking = df_metric_zscores.copy()
        df_z_for_ranking.columns = [
            col.replace("_Z", "") for col in df_z_for_ranking.columns
        ]

        df_metric_ranks = self.get_ranks(df_z_for_ranking)

        parts = [df, df_metric_zscores, df_metric_ranks]

        if include_pct_ranks:
            df_metric_pct_ranks = self.get_pct_ranks(df_z_for_ranking)
            parts.append(df_metric_pct_ranks)

        self.df = pd.concat(parts, axis=1)


class PlayerStats(Stats):
    """Player-level stats source."""

    data_point_class = data_point.Player
    negative_metrics = []

    def __init__(self, minimal_minutes: int = 300) -> None:
        """Initialize player stats.

        Args:
            minimal_minutes (int): Minimum minutes required for a player to be included.
        """
        self.minimal_minutes = minimal_minutes
        super().__init__()

    def get_raw_data(self) -> pd.DataFrame:
        """Load raw player data.

        Returns:
            pd.DataFrame: Raw player dataframe.
        """
        csv_path = working_dir / "data" / "events" / "Forwards.csv"
        return pd.read_csv(csv_path, encoding="unicode_escape")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate player data.

        Args:
            df_raw (pd.DataFrame): Raw player dataframe.

        Returns:
            pd.DataFrame: Processed player dataframe.
        """
        df_raw = df_raw.rename(columns={"shortName": "player_name"})
        df_raw = df_raw.replace({-1: np.nan})
        df_raw = df_raw[df_raw["Minutes"] >= self.minimal_minutes]

        if len(df_raw) < 10:
            raise Exception("Not enough players with enough minutes.")

        return df_raw

    def to_data_point(self, gender: str, position: str) -> data_point.Player:
        """Convert the currently filtered dataframe to a Player data point.

        Requirements:
        - self.df must be filtered to exactly one player row before calling.

        Args:
            gender (str): Gender label passed to the Player object.
            position (str): Position label passed to the Player object.

        Returns:
            data_point.Player: Player data point object.
        """
        id_val = self.df.index[0]

        self.df.reset_index(drop=True, inplace=True)

        name = self.df["player_name"][0]
        minutes_played = self.df["Minutes"][0]
        self.df = self.df.drop(columns=["player_name", "Minutes"])

        ser_metrics = self.df.squeeze()

        return self.data_point_class(
            id=id_val,
            name=name,
            minutes_played=minutes_played,
            gender=gender,
            position=position,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
        )


class CountryStats(Stats):
    """Country-level stats source with drill-down support."""

    data_point_class = data_point.Country
    negative_metrics = []

    def __init__(self) -> None:
        """Initialize country stats and drill-down metadata."""
        self.drill_down = self.get_drill_down_dict()
        self.drill_down_threshold = 1
        super().__init__()

    def get_drill_down_data(self, file_path: str | Path) -> dict:
        """Load and process drill-down z-score metadata for countries.

        Args:
            file_path (str | Path): Path to the input CSV.

        Returns:
            dict: Mapping from country to drill-down metric.
        """
        df = self.process_data(self.get_z_scores(pd.read_csv(file_path)))
        return dict(zip(df.country.values, df.drill_down_metric.values))

    def get_drill_down_data_values(
        self,
        file_path: str | Path,
        metric_name: str,
    ) -> dict:
        """Load drill-down raw value ranges for each country.

        Args:
            file_path (str | Path): Path to the input CSV.
            metric_name (str): Metric group name.

        Returns:
            dict: Mapping from country to a tuple of value ranges.
        """
        df = self.process_data(pd.read_csv(file_path))

        df["value_low"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][0]
            ],
            axis=1,
        )

        df["value_high"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][1]
            ],
            axis=1,
        )

        values = [
            (floor(l), ceil(h)) for l, h in zip(df["value_low"], df["value_high"])
        ]

        return dict(zip(df.country.values, values))

    def get_drill_down_dict(self) -> dict:
        """Build the drill-down metadata dictionary.

        Returns:
            dict: Nested drill-down dictionary by country and metric group.
        """
        path = working_dir / "data" / "wvs" / "intermediate_data"
        all_files = os.listdir(path)

        self.drill_down_metric_country_question = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data(path / file),
            )
            for file in all_files
            if file.endswith("_pre.csv")
        )

        drill_down_data_raw = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data_values(
                    path / file, "_".join(file.split("_")[:-1])
                ),
            )
            for file in all_files
            if file.endswith("_raw.csv")
        )

        metrics = [m for m in self.drill_down_metric_country_question.keys()]
        countries = [
            k for k in self.drill_down_metric_country_question[metrics[0]].keys()
        ]

        drill_down = [
            (
                country,
                dict(
                    [
                        (
                            metric,
                            (
                                self.drill_down_metric_country_question[metric][country],
                                drill_down_data_raw[metric][country],
                            ),
                        )
                        for metric in metrics
                    ]
                ),
            )
            for country in countries
        ]

        return dict(drill_down)

    def get_z_scores(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        negative_metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate z-scores and identify drill-down metrics.

        Args:
            df (pd.DataFrame): Input dataframe.
            metrics (list[str] | None): Metrics to include. If None, all except 'country'.
            negative_metrics (list[str] | None): Metrics to flip in z-score space.

        Returns:
            pd.DataFrame: Original dataframe concatenated with z-score columns
                and drill-down metric metadata.
        """
        if negative_metrics is None:
            negative_metrics = []

        if metrics is None:
            metrics = [m for m in df.columns if m not in ["country"]]

        df_z = df[metrics].apply(zscore, nan_policy="omit")
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        for metric in set(negative_metrics).intersection(metrics):
            z_col = f"{metric}_Z"
            if z_col in df_z.columns:
                df_z[z_col] = df_z[z_col] * -1

        drill_down_metrics_high = df[metrics].idxmax(axis=1)
        drill_down_metrics_low = df[metrics].idxmin(axis=1)
        drill_down_metrics = [
            (l, h) for l, h in zip(drill_down_metrics_low, drill_down_metrics_high)
        ]

        df_z["drill_down_metric"] = drill_down_metrics

        return pd.concat([df, df_z], axis=1)

    def select_random(self):
        """Return the index of a random sampled row.

        Returns:
            Any: Index of the sampled row.
        """
        return self.df.sample(1).index[0]

    def get_raw_data(self) -> pd.DataFrame:
        """Load raw country data.

        Returns:
            pd.DataFrame: Raw country dataframe.
        """
        csv_path = working_dir / "data" / "wvs" / "wave_7.csv"
        return pd.read_csv(csv_path)

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Validate country data.

        Args:
            df_raw (pd.DataFrame): Raw country dataframe.

        Returns:
            pd.DataFrame: Validated dataframe.
        """
        if df_raw["country"].isnull().values.any():
            raise ValueError("Country column contains NaN values.")

        if (df_raw["country"] == "").values.any():
            raise ValueError("Country column contains empty strings.")

        if df_raw["country"].duplicated().any():
            raise ValueError("Country column contains duplicates.")

        if len(df_raw) < 10:
            raise Exception("Not enough data points.")

        return df_raw

    def to_data_point(self) -> data_point.Country:
        """Convert the currently filtered dataframe to a Country data point.

        Requirements:
        - self.df must be filtered to exactly one country row before calling.

        Returns:
            data_point.Country: Country data point object.
        """
        id_val = self.df.index[0]

        self.df.reset_index(drop=True, inplace=True)

        name = self.df["country"][0]
        self.df = self.df.drop(columns=["country"])

        ser_metrics = self.df.squeeze()

        drill_down_metrics = ser_metrics[
            ser_metrics.index.str.endswith("_Z")
            & (ser_metrics.abs() >= self.drill_down_threshold)
        ].index.tolist()
        drill_down_metrics = [
            "_".join(x.split("_")[:-1]).lower() for x in drill_down_metrics
        ]

        drill_down_values = dict(
            [
                (key, value)
                for key, value in self.drill_down[name].items()
                if key.lower() in drill_down_metrics
            ]
        )

        return self.data_point_class(
            id=id_val,
            name=name,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
            drill_down_metrics=drill_down_values,
        )


class PersonStat(Stats):
    """Personality/person-level stats source."""

    data_point_class = data_point.Person
    negative_metrics = []

    def __init__(self) -> None:
        """Initialize person stats."""
        super().__init__()

    def get_raw_data(self) -> pd.DataFrame:
        """Load raw personality/person data.

        Returns:
            pd.DataFrame: Raw personality dataframe.
        """
        csv_path = working_dir / "data" / "data_raw.csv"
        return pd.read_csv(csv_path, sep="\t", encoding="unicode_escape")

    def get_questions(self) -> dict:
        """Return the question dictionary and scoring directions.

        Returns:
            dict: Mapping from question code to question text and coefficient.
        """
        ext_questions = {
            "EXT1": ["they are the life of the party", 1],
            "EXT2": ["they dont talk a lot", -1],
            "EXT3": ["they feel comfortable around people", 1],
            "EXT4": ["they keep in the background", -1],
            "EXT5": ["they start conversations", 1],
            "EXT6": ["they have little to say", -1],
            "EXT7": ["they talk to a lot of different people at parties", 1],
            "EXT8": ["they dont like to draw attention to themself", -1],
            "EXT9": ["they dont mind being the center of attention", 1],
            "EXT10": ["they are quiet around strangers", -1],
        }

        est_questions = {
            "EST1": ["they get stressed out easily", -1],
            "EST2": ["they are relaxed most of the time", 1],
            "EST3": ["they worry about things", -1],
            "EST4": ["they seldom feel blue", 1],
            "EST5": ["they are easily disturbed", -1],
            "EST6": ["they get upset easily", -1],
            "EST7": ["they change their mood a lot", -1],
            "EST8": ["they have frequent mood swings", -1],
            "EST9": ["they get irritated easily", -1],
            "EST10": ["they often feel blue", -1],
        }

        agr_questions = {
            "AGR1": ["they feel little concern for others", -1],
            "AGR2": ["they interested in people", 1],
            "AGR3": ["they insult people", -1],
            "AGR4": ["they sympathize with others feelings", 1],
            "AGR5": ["they are not interested in other peoples problems", -1],
            "AGR6": ["they have a soft heart", 1],
            "AGR7": ["they not really interested in others", -1],
            "AGR8": ["they take time out for others", 1],
            "AGR9": ["they feel others emotions", 1],
            "AGR10": ["they make people feel at ease", 1],
        }

        csn_questions = {
            "CSN1": ["they are always prepared", 1],
            "CSN2": ["they leave their belongings around", -1],
            "CSN3": ["they pay attention to details", 1],
            "CSN4": ["they make a mess of things", -1],
            "CSN5": ["they get chores done right away", 1],
            "CSN6": ["they often forget to put things back in their proper place", -1],
            "CSN7": ["they like order", 1],
            "CSN8": ["they shirk their duties", -1],
            "CSN9": ["they follow a schedule", 1],
            "CSN10": ["they are exacting in their work", 1],
        }

        opn_questions = {
            "OPN1": ["they have a rich vocabulary", 1],
            "OPN2": ["they have difficulty understanding abstract ideas", -1],
            "OPN3": ["they have a vivid imagination", 1],
            "OPN4": ["they are not interested in abstract ideas", -1],
            "OPN5": ["they have excellent ideas", 1],
            "OPN6": ["they do not have a good imagination", -1],
            "OPN7": ["they are quick to understand things", 1],
            "OPN8": ["they use difficult words", 1],
            "OPN9": ["they spend time reflecting on things", 1],
            "OPN10": ["they are full of ideas", 1],
        }

        questions = (
            ext_questions
            | est_questions
            | agr_questions
            | csn_questions
            | opn_questions
        )
        return questions

    def process_data(self, df_raw: pd.DataFrame | list) -> pd.DataFrame:
        """Process person data from a dataframe or from a list of answers.

        Args:
            df_raw (pd.DataFrame | list): Raw dataframe or a single answer list.

        Returns:
            pd.DataFrame: Processed personality dataframe.
        """
        questions = self.get_questions()

        if isinstance(df_raw, list):
            matching = [
                "EXT1", "EXT2", "EXT3", "EXT4", "EXT5",
                "EXT6", "EXT7", "EXT8", "EXT9", "EXT10",
                "EST1", "EST2", "EST3", "EST4", "EST5",
                "EST6", "EST7", "EST8", "EST9", "EST10",
                "AGR1", "AGR2", "AGR3", "AGR4", "AGR5",
                "AGR6", "AGR7", "AGR8", "AGR9", "AGR10",
                "CSN1", "CSN2", "CSN3", "CSN4", "CSN5",
                "CSN6", "CSN7", "CSN8", "CSN9", "CSN10",
                "OPN1", "OPN2", "OPN3", "OPN4", "OPN5",
                "OPN6", "OPN7", "OPN8", "OPN9", "OPN10",
            ]
            df_raw = pd.DataFrame([df_raw], columns=[column for column in matching])

        else:
            df_raw = df_raw.copy()
            df_raw.drop(df_raw.columns[50:107], axis=1, inplace=True)
            df_raw.drop(df_raw.columns[50:], axis=1, inplace=True)
            df_raw.dropna(inplace=True)

        for column in df_raw.columns:
            df_raw[column] = df_raw[column] * questions[column][1]

        df_raw["extraversion"] = df_raw.iloc[:, 0:10].sum(axis=1) + 20
        df_raw["neuroticism"] = df_raw.iloc[:, 10:20].sum(axis=1) + 38
        df_raw["agreeableness"] = df_raw.iloc[:, 20:30].sum(axis=1) + 14
        df_raw["conscientiousness"] = df_raw.iloc[:, 30:40].sum(axis=1) + 14
        df_raw["openness"] = df_raw.iloc[:, 40:50].sum(axis=1) + 8
        df_raw["name"] = df_raw.index.to_series().apply(lambda idx: "C_" + str(idx))

        return df_raw

    def to_data_point(self) -> data_point.Person:
        """Convert the currently filtered dataframe to a Person data point.

        Requirements:
        - self.df must be filtered to exactly one person row before calling.

        Returns:
            data_point.Person: Person data point object.
        """
        id_val = self.df.index[0]
        name = self.df["name"].values[0]

        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.drop(columns=["name"])

        ser_metrics = self.df.squeeze()

        return self.data_point_class(id=id_val, name=name, ser_metrics=ser_metrics)


class TeamStats(Stats):
    """Team-level stats source for the team build-up analyst.

    This class automatically finds a CSV file in the configured folder,
    so the exact file name does not matter as long as there is exactly
    one CSV file in that folder.
    """

    data_point_class = data_point.Team

    def __init__(
        self,
        data_folder: str | Path = "data/team_build_up_analyst",
        minimal_minutes: int = 0,
        file_pattern: str = "*.csv",
    ) -> None:
        """Initialize team stats.

        Args:
            data_folder (str | Path): Folder containing the team CSV file.
            minimal_minutes (int): Minimum 'total_minutes' threshold if the
                column exists.
            file_pattern (str): Pattern used to search for files.

        Raises:
            FileNotFoundError: If the data folder or file is missing.
            ValueError: If multiple matching CSV files are found.
        """
        self.data_folder = working_dir / Path(data_folder)
        self.minimal_minutes = minimal_minutes
        self.file_pattern = file_pattern
        self.csv_path = self._resolve_csv_path()
        super().__init__()

    def _resolve_csv_path(self) -> Path:
        """Resolve the CSV file automatically from the data folder.

        Returns:
            Path: Resolved CSV file path.

        Raises:
            FileNotFoundError: If no CSV file is found.
            ValueError: If multiple CSV files are found.
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder does not exist: {self.data_folder}")

        csv_files = sorted(self.data_folder.glob(self.file_pattern))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {self.data_folder}")

        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple CSV files found in {self.data_folder}. "
                f"Expected exactly one file, found: {[file.name for file in csv_files]}"
            )

        return csv_files[0]

    def get_raw_data(self) -> pd.DataFrame:
        """Load raw team data from the resolved CSV file.

        Returns:
            pd.DataFrame: Raw team dataframe.
        """
        return pd.read_csv(self.csv_path, encoding="utf-8")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean team data.

        Required columns:
        - team

        Optional columns:
        - total_minutes

        Args:
            df_raw (pd.DataFrame): Raw team dataframe.

        Returns:
            pd.DataFrame: Processed team dataframe.
        """
        required_cols = {"team"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df_raw.copy()
        df = df.replace({-1: np.nan})

        df = df.dropna(subset=["team"])
        df["team"] = df["team"].astype(str).str.strip()
        df = df[df["team"] != ""]

        if self.minimal_minutes > 0 and "total_minutes" in df.columns:
            df = df[df["total_minutes"] >= self.minimal_minutes]

        if len(df) < 5:
            raise Exception("Not enough teams to plot or rank.")

        return df.reset_index(drop=True)

    def get_available_metrics(
        self,
        exclude_columns: Iterable[str] | None = None,
        numeric_only: bool = True,
    ) -> list[str]:
        """Return available metric columns for team data.

        Args:
            exclude_columns (Iterable[str] | None): Columns to exclude.
            numeric_only (bool): If True, only return numeric columns.

        Returns:
            list[str]: List of available metric names.
        """
        if exclude_columns is None:
            exclude_columns = {"team", "total_minutes"}
        else:
            exclude_columns = set(exclude_columns)

        if numeric_only:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            return [col for col in numeric_cols if col not in exclude_columns]

        return [col for col in self.df.columns if col not in exclude_columns]

    def to_data_point(self) -> data_point.Team:
        """Convert the currently filtered dataframe to a Team data point.

        Requirements:
        - self.df must be filtered to exactly one team row before calling.

        Returns:
            data_point.Team: Team data point object.
        """
        if len(self.df) != 1:
            raise ValueError(
                "TeamStats.to_data_point() expects self.df to contain exactly one row."
            )

        id_val = self.df.index[0]
        df_one = self.df.reset_index(drop=True)

        name = df_one.loc[0, "team"]
        minutes_played = (
            df_one.loc[0, "total_minutes"] if "total_minutes" in df_one.columns else None
        )

        drop_cols = ["team"]
        if "total_minutes" in df_one.columns:
            drop_cols.append("total_minutes")

        df_metrics = df_one.drop(columns=drop_cols, errors="ignore")
        ser_metrics = df_metrics.squeeze()

        return self.data_point_class(
            id=id_val,
            name=name,
            minutes_played=minutes_played,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
        )

    def to_data_point_by_team(self, team_name: str) -> data_point.Team:
        """Create a Team data point for one selected team without mutating self.df.

        Args:
            team_name (str): Team name exactly as it appears in the dataframe.

        Returns:
            data_point.Team: Team data point object.

        Raises:
            ValueError: If the team is not found.
        """
        df_one = self.df[self.df["team"] == team_name].copy()

        if df_one.empty:
            raise ValueError(f"Team '{team_name}' not found in dataset.")

        id_val = df_one.index[0]
        df_one.reset_index(drop=True, inplace=True)

        name = df_one.loc[0, "team"]
        minutes_played = (
            df_one.loc[0, "total_minutes"] if "total_minutes" in df_one.columns else None
        )

        drop_cols = ["team"]
        if "total_minutes" in df_one.columns:
            drop_cols.append("total_minutes")

        df_metrics = df_one.drop(columns=drop_cols, errors="ignore")
        ser_metrics = df_metrics.squeeze()

        return self.data_point_class(
            id=id_val,
            name=name,
            minutes_played=minutes_played,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
        )