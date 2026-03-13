import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from utils.page_components import add_common_page_elements
from classes.data_source import TeamStats
from classes.visual import DistributionPlot


def main():
    """Run the Team Build-Up Analyst page."""
    add_common_page_elements()

    st.title("Team Build-Up Analyst")
    st.write("Team build-up profiling using tournament-level metrics.")

    metrics = [
        "buildup_to_create_pct",
        "buildup_to_direct_pct",
        "buildup_that_ends_with_finish_pct",
        "turnover_pct_buildup",
        "opp_box_entries_within_7s_after_turnover",
        "opp_shot_probability_within_7s_after_turnover",
        "first_line_break_pct_buildup",
        # "second_last_line_break_pct_buildup",
    ]

    negative_metrics = [
        "turnover_pct_buildup",
        "opp_box_entries_within_7s_after_turnover",
        "opp_shot_probability_within_7s_after_turnover",
    ]

    # TeamStats now resolves the CSV automatically from the folder.
    teams_all = TeamStats(
        data_folder="data/team_build_up_analyst"
    )
    teams_all.calculate_statistics(
        metrics=metrics,
        negative_metrics=negative_metrics
    )

    with st.sidebar:
        st.subheader("Team selection")
        selected_team = st.selectbox(
            "Select team",
            options=teams_all.df["team"].unique(),
            index=0,
        )

    team = teams_all.to_data_point_by_team(selected_team)

    plot = DistributionPlot(
        columns=metrics,
        labels=["Worse", "Average", "Better"],
        plot_type="default",
    )

    plot.add_title(
        title=f"Team build-up profile: {team.name}",
        subtitle="Compared to other teams in the dataset (z-scores)",
    )

    plot.add_players(players=teams_all, metrics=metrics)
    plot.add_player(player=team, n_group=len(teams_all.df), metrics=metrics)

    plot.show()

    st.subheader("Correlation Matrix")

    corr_df = teams_all.df[metrics].corr(method="pearson")

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        ["#d73027", "#ffffff", "#1a9850"],
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_df,
        annot=True,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Correlation Matrix – Team Build-Up Metrics")

    st.pyplot(fig)


if __name__ == "__main__":
    main()