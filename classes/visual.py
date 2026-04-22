import streamlit as st
import plotly.graph_objects as go
import numpy as np

from typing import Union

from utils.sentences import format_metric
from classes.data_point import Player, Country, Person, Team
from classes.data_source import PlayerStats, CountryStats, PersonStat, TeamStats


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity: float = 1):
    """Convert RGB tuple to rgba() string."""
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color: str, text: str, alpha: float = 1.0) -> str:
    """Return colored HTML span for axis tick labels."""
    return (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )


class Visual:
    """Base class for Plotly visuals used in the app."""

    dark_green = hex_to_rgb("#002c1c")
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb("#00A938")
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")

    def __init__(self, pdf: bool = False, plot_type: str = "scout"):
        self.pdf = pdf
        self.font_size_multiplier = 1.4 if pdf else 1.0

        self.fig = go.Figure()
        self._setup_styles()
        self.plot_type = plot_type

        if plot_type == "scout":
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f} per 90</span>"
        else:
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def _setup_styles(self):
        """Apply shared layout styles."""
        side_margin = 60
        top_margin = 75
        pad = 16

        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title: str, subtitle: str):
        """Add chart title and subtitle."""
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text: str):
        """Add annotation centered below the chart."""
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )

    def show(self):
        """Render Plotly chart in Streamlit."""
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            width="content",
            key=f"plotly_{id(self)}"
        )

    def close(self):
        """Placeholder for compatibility."""
        pass


class DistributionPlot(Visual):
    """Distribution plot used for players, teams, and countries."""

    def __init__(self, columns, labels=None, display_names=None, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.display_names = display_names or {}

        self.marker_color = (c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])

        super().__init__(*args, **kwargs)

        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()
            
    def _setup_axes(self, labels=("Worse", "Average", "Better")):
        """Set x/y axes for distribution plot."""
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=list(labels),
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

        self.fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=len(self.columns),
            line=dict(color="gray", width=1, dash="dot"),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover=""):
        """
        Add all group points as faint markers.
        Hover shows name, raw value, and rank.
        """
        group_size = len(df_plot)

        for i, col in enumerate(self.columns):
            hovertext = [
                f"{name}<br>Value: {value:.2f}<br>Rank: {int(rank)}/{group_size}"
                for name, value, rank in zip(
                    names,
                    df_plot[col].tolist(),
                    df_plot[col + hover].tolist(),
                )
            ]

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots].tolist(),
                    y=list(np.ones(len(df_plot[col + plots])) * i),
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.dark_green, opacity=0.2),
                        "size": 10,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(self.bright_green),
                    },
                    hovertext=hovertext,
                    hovertemplate="%{hovertext}<extra></extra>",
                    text=names,
                    showlegend=False,
                )
            )

    def add_data_point(self, ser_plot, plots, name, hover="", text=None, n_group=None):
        """
        Add one focal point with distinct marker style.
        Hover shows name, raw value, and rank.
        """
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            metric_name = self.display_names.get(col, format_metric(col))

            rank_val = int(ser_plot[col + hover])
            if n_group is not None:
                hovertext = [f"{name}<br>Value: {ser_plot[col]:.2f}<br>Rank: {rank_val}/{n_group}"]
            else:
                hovertext = [f"{name}<br>Value: {ser_plot[col]:.2f}<br>Rank: {rank_val}"]

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertext=hovertext,
                    hovertemplate="%{hovertext}<extra></extra>",
                    text=text,
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=self.annotation_text.format(metric_name=metric_name, data=ser_plot[col]),
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_player(self, player: Union[Player, Country, Team], n_group: int, metrics):
        """Add the focal player/team/country point."""
        if isinstance(player, (Player, Country, Team)):
            ser_plot = player.ser_metrics
            name = player.name
        else:
            raise TypeError("Invalid type: expected Player, Country, or Team")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            n_group=n_group,
        )

    def add_players(self, players: Union[PlayerStats, CountryStats, TeamStats], metrics):
        """Add the background group distribution."""
        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                legend="Other players  ",
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                legend="Other countries  ",
            )
        elif isinstance(players, TeamStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["team"],
                hover="_Ranks",
                legend="Other teams  ",
            )
        else:
            raise TypeError("Invalid type: expected PlayerStats, CountryStats, or TeamStats")

    def add_title_from_player(self, player: Union[Player, Country, Team]):
        """Set title/subtitle from a datapoint."""
        self.player = player
        title = f"Evaluation of {player.name}?"

        if isinstance(player, (Player, Team)):
            if getattr(player, "minutes_played", None) is not None:
                subtitle = f"Based on {player.minutes_played} minutes played"
            else:
                subtitle = "Based on the available dataset"
        elif isinstance(player, Country):
            subtitle = "Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid type: expected Player, Country, or Team")

        self.add_title(title, subtitle)


class DistributionPlotPersonality(Visual):
    """Distribution plot for personality test page."""

    def __init__(self, columns, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=["Worse", "Average", "Better"],
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover=""):
        """Add all group points with hover showing value and rank."""
        group_size = len(df_plot)

        for i, col in enumerate(self.columns):
            hovertext = [
                f"{name}<br>Value: {value:.2f}<br>Rank: {int(rank)}/{group_size}"
                for name, value, rank in zip(
                    names,
                    df_plot[col].tolist(),
                    df_plot[col + hover].tolist(),
                )
            ]

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots].tolist(),
                    y=list(np.ones(len(df_plot[col + plots])) * i),
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.dark_green, opacity=0.2),
                        "size": 10,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(self.bright_green),
                    },
                    hovertext=hovertext,
                    hovertemplate="%{hovertext}<extra></extra>",
                    text=names,
                    showlegend=False,
                )
            )

    def add_data_point(self, ser_plot, plots, name, hover="", text=None, n_group=None):
        """Add one focal personality point."""
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            metric_name = format_metric(col)

            rank_val = int(ser_plot[col + hover])
            if n_group is not None:
                hovertext = [f"{name}<br>Value: {ser_plot[col]:.2f}<br>Rank: {rank_val}/{n_group}"]
            else:
                hovertext = [f"{name}<br>Value: {ser_plot[col]:.2f}<br>Rank: {rank_val}"]

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertext=hovertext,
                    hovertemplate="%{hovertext}<extra></extra>",
                    text=text,
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_person(self, person: Person, n_group, metrics):
        """Add focal person."""
        self.add_data_point(
            ser_plot=person.ser_metrics,
            plots="_Z",
            name=person.name,
            hover="_Ranks",
            n_group=n_group,
        )

    def add_persons(self, persons: PersonStat, metrics):
        """Add all persons."""
        self.add_group_data(
            df_plot=persons.df,
            plots="_Z",
            names=persons.df["name"],
            hover="_Ranks",
            legend="Other persons  ",
        )

    def add_title_from_person(self, person: Person):
        """Set title/subtitle for personality plot."""
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = "Based on Big Five scores"
        self.add_title(title, subtitle)
