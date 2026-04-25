"""
Entrypoint for the Team Build-Up Analyst Streamlit page.
Runs top to bottom every time the user interacts with the app.
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from classes.data_source import TeamStats
from classes.visual import DistributionPlot
from classes.description import TeamDescription
from classes.chat import TeamChat

from utils.page_components import add_common_page_elements
from utils.utils import create_chat


# -------------------------------
# Page setup
# -------------------------------
sidebar_container = add_common_page_elements()
st.divider()

st.title("Team Build-Up Analyst")
st.write(
    "Explore how teams build up play and how effective they are. "
    "Ask about a team to analyse their style, strengths, and weaknesses."
)

# -------------------------------
# Metrics
# -------------------------------
metrics = [
    "progression_to_midfield_pct",
    "buildup_that_ends_with_finish_pct",
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
    "first_line_break_pct_buildup",
    # STYLE
    "prop_gk_involved",
    "avg_passes",
    "avg_duration",
    "avg_players_involved",
    "build_ups_per_game",
    "prop_channel_center",
    "prop_channel_half_space_left",
    "prop_channel_wide_left",
    "prop_channel_half_space_right",
    "prop_channel_wide_right",
]

style_metrics = [
    "prop_gk_involved",
    "avg_passes",
    "avg_duration",
    "avg_players_involved",
    "build_ups_per_game",
    "prop_channel_center",
    "prop_channel_half_space_left",
    "prop_channel_wide_left",
    "prop_channel_half_space_right",
    "prop_channel_wide_right",
]

quality_metrics = [
    "first_line_break_pct_buildup",
    "progression_to_midfield_pct",
    "buildup_that_ends_with_finish_pct",
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
]

negative_metrics = [
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
]

# -------------------------------
# Load dataset
# -------------------------------
teams = TeamStats(data_folder="data/team_build_up_analyst")

teams.calculate_statistics(
    metrics=metrics,
    negative_metrics=negative_metrics,
)

# -------------------------------
# Show dataset
# -------------------------------
st.expander("Dataframe used", expanded=False).write(teams.df)

# -------------------------------
# Chat setup (NO TEAM INITIALLY)
# -------------------------------
to_hash = ("no_team_selected", "team_build_up_analyst")
chat = create_chat(to_hash, TeamChat, None, teams)

# -------------------------------
# Initial chat state
# -------------------------------
if chat.state == "empty":

    sample_teams = teams.df["team"].dropna().unique().tolist()[:8]
    examples = ", ".join(sample_teams)

    chat.add_message(
        (
            "Which team would you like to analyse today?\n\n"
            f"You can type any team name (for example: {examples}). "
            "Once selected, I can break down their build-up style, performance, strengths, "
            "and compare them to other teams."
        ),
        role="assistant",
    )

    chat.state = "awaiting_team"

# -------------------------------
# Chat interaction
# -------------------------------
chat.get_input()
chat.display_messages()
chat.save_state()