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


# Add common page elements and create sidebar containers
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

st.title("Team Build-Up Analyst")
st.write("Team build-up analyst measuring teams performance in their build-up play. Performance is related to ball progression, turnover and opposition chances during build-up.")

# -------------------------------
# Metrics
# -------------------------------

# All metrics
metrics = [
    #"buildup_to_create_pct",
    #"buildup_to_direct_pct",
    "progression_to_midfield_pct",
    "buildup_that_ends_with_finish_pct",
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
    "first_line_break_pct_buildup",
    "prop_gk_involved", # STYLE STARTS HERE
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

# Split into STYLE vs QUALITY
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

# Metrics where lower values are better
negative_metrics = [
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
]

# -------------------------------
# Load team dataset
# -------------------------------
teams = TeamStats(data_folder="data/team_build_up_analyst")

teams.calculate_statistics(
    metrics=metrics,
    negative_metrics=negative_metrics,
)

# -------------------------------
# Team selection
# -------------------------------
with sidebar_container:
    st.subheader("Team selection")
    selected_team = st.selectbox(
        "Select team",
        options=teams.df["team"].unique(),
        index=0,
    )

team = teams.to_data_point_by_team(selected_team)

# Optional: show dataset
st.expander("Dataframe used", expanded=False).write(teams.df)

# -------------------------------
# Chat setup
# -------------------------------
to_hash = (team.id, "team_build_up_analyst")
chat = create_chat(to_hash, TeamChat, team, teams)

# -------------------------------
# Initial chat content
# -------------------------------
if chat.state == "empty":

    # Map each metric to a custom display name
    display_names = {
        "buildup_to_create_pct": "Buildup to Create (%)",
        "buildup_to_direct_pct": "Buildup to Direct (%)",
        "progression_to_midfield_pct": "Progression to Midfield (%)",
        "buildup_that_ends_with_finish_pct": "Buildup Ending in Finish Phase (%)",
        "first_line_break_pct_buildup": "1st Line Break (%)",
        "second_last_line_break_pct_buildup": "2nd Last Line Break (%)",
        "turnover_pct_buildup": "Turnover (%)",
        "opp_box_entries_within_7s_after_turnover": "Opp. Box Entries in 7s after Turnover",
        "opp_shot_probability_within_7s_after_turnover": "Opp. Shot Prob in 7s after Turnover",
        "prop_gk_involved": "Goalkeeper Involved (%)",
        "avg_passes": "Avg Passes (#)",
        "avg_duration": "Avg Duration (s)",
        "avg_players_involved": "Avg Players Involved",
        "build_ups_per_game": "Build-Ups per Game",
        "prop_channel_center": "Central Channel (%)",
        "prop_channel_half_space_left": "Left Half-Space (%)",
        "prop_channel_wide_left": "Left Wide (%)",
        "prop_channel_half_space_right": "Right Half-Space (%)",
        "prop_channel_wide_right": "Right Wide (%)"
    }

    # STYLE PLOT
#    style_plot = DistributionPlot(
#        columns=style_metrics[::-1],
#        labels=["Worse", "Average", "Better"],
#        plot_type="default",
#    )
#    style_plot.add_title(
#        title=f"{team.name} – Build-Up Style",
#        subtitle="How the team builds up play (z-scores)",
#    )
#    style_plot.add_players(teams, metrics=style_metrics)
#    style_plot.add_player(team, len(teams.df), metrics=style_metrics)

    # QUALITY PLOT
    quality_plot = DistributionPlot(
        columns=quality_metrics[::-1],
        labels=["Worse", "Average", "Better"],
        plot_type="default",
        display_names=display_names  
    )

    quality_plot.add_title(
        title=f"{team.name} – Build-Up Quality",
        subtitle="Effectiveness and outcomes of build-up (z-scores)",
    )
    quality_plot.add_players(teams, metrics=quality_metrics)
    quality_plot.add_player(team, len(teams.df), metrics=quality_metrics)


    # Team summary
    description = TeamDescription(team)
    summary = description.stream_gpt(stream=True)

    # Add to chat
    chat.add_message(
        f"Please can you summarise {team.name}'s build-up for me?",
        role="user",
        user_only=False,
        visible=False,
    )
#    chat.add_message(style_plot)
    chat.add_message(quality_plot)
    chat.add_message(summary)

    chat.state = "default"

# -------------------------------
# Chat interaction
# -------------------------------
chat.get_input()
chat.display_messages()
chat.save_state()

# -------------------------------
# Correlation matrix (MOVED HERE)
# -------------------------------
show_correlation_matrix = False

if show_correlation_matrix:
    st.subheader("Correlation Matrix")

    corr_df = teams.df[metrics].corr(method="pearson")

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        ["#d73027", "#ffffff", "#1a9850"],
    )

    sns.set_style("white")
    sns.set_context("notebook", font_scale=0.8)

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="none")

    def format_label(label):
        has_pct = False
        if "pct" in label:
            has_pct = True
            label = label.replace("pct", "").strip("_")
        label = label.replace("_", " ").title()
        if has_pct:
            label += " (%)"
        if len(label) > 20:
            parts = label.split(" ")
            mid = len(parts) // 2
            label = " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
        return label

    labels = [format_label(m) for m in metrics]

    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="lightgray",
        square=True,
        annot_kws={"size": 8, "weight": "bold"},
        cbar_kws={"shrink": 0.7},
    )

    ax.set_xticklabels(labels, rotation=45, ha="right", wrap=True)
    ax.set_yticklabels(labels, rotation=0)

    sns.despine(left=True, bottom=True)

    ax.set_title(
        "Correlation Matrix – Team Build-Up Metrics",
        fontsize=12,
        weight="bold",
    )

    st.pyplot(fig, transparent=True)