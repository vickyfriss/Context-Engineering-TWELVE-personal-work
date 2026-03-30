import pandas as pd
from pathlib import Path
from collections import defaultdict


# Working directory
WD = Path.cwd()

# Input / output paths
DYNAMIC_DIR = Path("../data/dynamic_events_pl_24/dynamic")
OUTPUT_PATH = Path("../data/team_build_up_analyst/buildup_metrics.csv")

# 10 FPS -> 7 seconds = 70 frames
SEVEN_SECONDS_FRAMES = 70

# Possession chains are stopped if one of these phases appears
OUT_OF_PLAY_PHASES = {"set_play", "disruption"}


def time_to_seconds(t):
    """
    Convert a mm:ss.t-like time format to seconds.

    Parameters
    ----------
    t : Any
        Time value expected to be in a string-like format such as '12:34.5'.

    Returns
    -------
    float or pd.NA
        Time converted to seconds. Returns pd.NA if parsing fails.

    Notes
    -----
    This function assumes the time is within a single period and that the
    decimal part represents tenths of a second.
    """
    if pd.isna(t):
        return pd.NA

    t = str(t).strip()
    if t == "":
        return pd.NA

    parts = t.split(":")
    if len(parts) != 2:
        return pd.NA

    try:
        minutes = int(parts[0])
        sec_parts = parts[1].split(".")
        seconds = int(sec_parts[0])
        tenths = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return minutes * 60 + seconds + tenths / 10
    except Exception:
        return pd.NA


def build_phase_data(df):
    """
    Build a lookup structure with phase-level information from player possession events.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dynamic events DataFrame.

    Returns
    -------
    tuple
        phase_data : dict
            Mapping from phase_index to:
            - team
            - phase_type
            - had_turnover
        sorted_indices : list
            Sorted list of phase_index values.
        pos_lookup : dict
            Mapping from phase_index to its position in sorted_indices.

    Requirements
    ------------
    The DataFrame must include at least:
    - event_type
    - phase_index
    - team_shortname
    - team_in_possession_phase_type
    - team_possession_loss_in_phase
    """
    phase_data = {}

    pp_df = df[df["event_type"] == "player_possession"]

    for phase_idx, grp in pp_df.groupby("phase_index"):
        phase_data[phase_idx] = {
            "team": grp["team_shortname"].iloc[0],
            "phase_type": grp["team_in_possession_phase_type"].iloc[0],
            "had_turnover": bool(grp["team_possession_loss_in_phase"].any()),
        }

    sorted_indices = sorted(phase_data.keys())
    pos_lookup = {idx: i for i, idx in enumerate(sorted_indices)}

    return phase_data, sorted_indices, pos_lookup


def chain_reaches_finish(start_phase_idx, team, phase_data, sorted_indices, pos_lookup):
    """
    Check whether a build-up phase continues into a possession chain that reaches finish.

    Parameters
    ----------
    start_phase_idx : int
        The phase_index of the starting build-up phase.
    team : str
        Team shortname.
    phase_data : dict
        Output from build_phase_data().
    sorted_indices : list
        Sorted phase indices.
    pos_lookup : dict
        Mapping from phase_index to position.

    Returns
    -------
    bool
        True if the possession chain reaches a later 'finish' phase without:
        - turnover,
        - change of team in possession,
        - entering set_play or disruption.
    """
    if start_phase_idx not in pos_lookup:
        return False

    if phase_data[start_phase_idx]["had_turnover"]:
        return False

    pos = pos_lookup[start_phase_idx] + 1

    while pos < len(sorted_indices):
        idx = sorted_indices[pos]
        info = phase_data[idx]

        if info["team"] != team:
            return False

        if info["phase_type"] in OUT_OF_PLAY_PHASES:
            return False

        if info["phase_type"] == "finish":
            return True

        if info["had_turnover"]:
            return False

        pos += 1

    return False


def compute_match_team_metrics(df, team, phase_data, sorted_indices, pos_lookup):
    """
    Compute build-up metrics for one team in one match.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dynamic events DataFrame.
    team : str
        Team shortname.
    phase_data : dict
        Output from build_phase_data().
    sorted_indices : list
        Sorted phase indices.
    pos_lookup : dict
        Mapping from phase_index to position.

    Returns
    -------
    dict or None
        Dictionary with aggregated match metrics for the team, or None if the
        team has no build-up player possession events in the match.

    Requirements
    ------------
    The DataFrame must include the columns used in the filtering, grouping, and
    calculations below.
    """
    pp_bu = df[
        (df["event_type"] == "player_possession")
        & (df["team_in_possession_phase_type"] == "build_up")
        & (df["team_shortname"] == team)
    ]

    if pp_bu.empty:
        return None

    phase_summary = (
        pp_bu.groupby("phase_index", sort=True)
        .agg(
            had_turnover=("team_possession_loss_in_phase", "first"),
            next_phase=("current_team_in_possession_next_phase_type", "first"),
            last_frame=("frame_start", "max"),
            last_period=("period", "last"),
            goalkeeper_involved=("player_in_possession_position", lambda s: (s == "GK").any()),
            successful_passes=("pass_outcome", lambda s: (s == "successful").sum()),
            start_time=("time_start", "first"),
            end_time=("time_end", "last"),
            players_involved=("player_in_possession_id", lambda s: s.dropna().nunique()),
            last_channel=("channel_start", "last"),
        )
        .reset_index()
    )

    if phase_summary.empty:
        return None

    # Convert time strings to seconds and calculate phase duration
    phase_summary["start_seconds"] = phase_summary["start_time"].apply(time_to_seconds)
    phase_summary["end_seconds"] = phase_summary["end_time"].apply(time_to_seconds)
    phase_summary["duration"] = phase_summary["end_seconds"] - phase_summary["start_seconds"]

    n_phases = len(phase_summary)
    n_turnovers = int((phase_summary["had_turnover"] == True).sum())

    no_turnover_mask = phase_summary["had_turnover"] == False
    next_phase = phase_summary["next_phase"]

    n_to_create = int(((next_phase == "create") & no_turnover_mask).sum())
    n_to_direct = int(((next_phase == "direct") & no_turnover_mask).sum())
    n_progress_to_midfield = int((next_phase.isin(["create", "direct"]) & no_turnover_mask).sum())

    n_chain_to_finish = sum(
        1
        for phase_idx in phase_summary["phase_index"]
        if chain_reaches_finish(phase_idx, team, phase_data, sorted_indices, pos_lookup)
    )

    first_lb_phases = set(pp_bu[pp_bu["first_line_break"] == True]["phase_index"].unique())
    second_lb_phases = set(pp_bu[pp_bu["second_last_line_break"] == True]["phase_index"].unique())

    n_first_lb = sum(1 for phase_idx in phase_summary["phase_index"] if phase_idx in first_lb_phases)
    n_second_lb = sum(1 for phase_idx in phase_summary["phase_index"] if phase_idx in second_lb_phases)

    turnover_phases = phase_summary[phase_summary["had_turnover"] == True]

    total_box_entries = 0
    total_xshot_max = 0.0

    for _, row in turnover_phases.iterrows():
        frame_t = row["last_frame"]
        period = row["last_period"]

        window = df[
            (df["frame_start"] > frame_t)
            & (df["frame_start"] <= frame_t + SEVEN_SECONDS_FRAMES)
            & (df["period"] == period)
        ]

        # Opponent box entries after build-up turnover
        opponent_box_possessions = window[
            (window["event_type"] == "player_possession")
            & (window["team_shortname"] != team)
            & (window["penalty_area_start"] == True)
        ]
        total_box_entries += len(opponent_box_possessions)

        # Maximum opponent shot probability during the next sequence
        pressing_team_obe = window[
            (window["event_type"] == "on_ball_engagement")
            & (window["team_shortname"] == team)
        ]
        xshot_vals = pressing_team_obe["xshot_player_possession_max"].dropna()
        total_xshot_max += xshot_vals.max() if len(xshot_vals) > 0 else 0.0

    # Additional phase-level accumulation
    sum_passes = phase_summary["successful_passes"].sum()
    sum_duration = phase_summary["duration"].sum()
    sum_players = phase_summary["players_involved"].sum()
    sum_gk = phase_summary["goalkeeper_involved"].sum()

    channel_counts = phase_summary["last_channel"].value_counts()

    channel_dict = {}
    for channel_name, count in channel_counts.items():
        channel_dict[f"channel_{channel_name}"] = count

    return {
        "n_phases": n_phases,
        "n_turnovers": n_turnovers,
        "n_to_create": n_to_create,
        "n_to_direct": n_to_direct,
        "n_progress_to_midfield": n_progress_to_midfield,
        "n_chain_to_finish": n_chain_to_finish,
        "n_first_lb": n_first_lb,
        "n_second_lb": n_second_lb,
        "box_entries": total_box_entries,
        "xshot_max": total_xshot_max,
        "sum_passes": sum_passes,
        "sum_duration": sum_duration,
        "sum_players": sum_players,
        "sum_gk": sum_gk,
        **channel_dict,
    }


def main():
    """
    Process all match parquet files and create a team-level build-up metrics CSV.

    Returns
    -------
    None

    Output
    ------
    Saves a CSV file to OUTPUT_PATH with one row per team.
    """
    parquet_files = sorted(DYNAMIC_DIR.glob("*.parquet"))

    print(f"Working directory: {WD}")
    print(f"Processing {len(parquet_files)} match files...")

    team_totals = defaultdict(lambda: defaultdict(float))
    team_matches = defaultdict(set)

    for i, file_path in enumerate(parquet_files, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(parquet_files)}")

        df = pd.read_parquet(file_path)

        phase_data, sorted_indices, pos_lookup = build_phase_data(df)

        teams = df[df["event_type"] == "player_possession"]["team_shortname"].dropna().unique()

        for team in teams:
            result = compute_match_team_metrics(
                df=df,
                team=team,
                phase_data=phase_data,
                sorted_indices=sorted_indices,
                pos_lookup=pos_lookup,
            )

            if result is not None:
                for key, value in result.items():
                    team_totals[team][key] += value

                team_matches[team].add(file_path.name)

    rows = []

    for team, totals in team_totals.items():
        n_phases = totals["n_phases"]
        n_turnovers = totals["n_turnovers"]
        n_matches = len(team_matches[team])

        row = {
            "team": team,
            "buildup_to_create_pct": round(100 * totals["n_to_create"] / n_phases, 2) if n_phases > 0 else 0,
            "buildup_to_direct_pct": round(100 * totals["n_to_direct"] / n_phases, 2) if n_phases > 0 else 0,
            "progression_to_midfield_pct": round(100 * totals["n_progress_to_midfield"] / n_phases, 2) if n_phases > 0 else 0,
            "buildup_that_ends_with_finish_pct": round(100 * totals["n_chain_to_finish"] / n_phases, 2) if n_phases > 0 else 0,
            "first_line_break_pct_buildup": round(100 * totals["n_first_lb"] / n_phases, 2) if n_phases > 0 else 0,
            "second_last_line_break_pct_buildup": round(100 * totals["n_second_lb"] / n_phases, 2) if n_phases > 0 else 0,
            "turnover_pct_buildup": round(100 * n_turnovers / n_phases, 2) if n_phases > 0 else 0,
            "opp_box_entries_within_7s_after_turnover": round(totals["box_entries"] / n_turnovers, 4) if n_turnovers > 0 else 0,
            "opp_shot_probability_within_7s_after_turnover": round(totals["xshot_max"] / n_turnovers, 4) if n_turnovers > 0 else 0,
            "prop_gk_involved": round(totals["sum_gk"] / n_phases, 3) if n_phases > 0 else 0,
            "avg_passes": round(totals["sum_passes"] / n_phases, 2) if n_phases > 0 else 0,
            "avg_duration": round(totals["sum_duration"] / n_phases, 2) if n_phases > 0 else 0,
            "avg_players_involved": round(totals["sum_players"] / n_phases, 2) if n_phases > 0 else 0,
            "build_ups_per_game": round(n_phases / n_matches, 2) if n_matches > 0 else 0,
        }

        # Add channel proportions dynamically
        channel_cols = [key for key in totals.keys() if key.startswith("channel_")]
        for channel_col in channel_cols:
            row[f"prop_{channel_col}"] = round(totals[channel_col] / n_phases, 3) if n_phases > 0 else 0

        rows.append(row)

    output_df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(output_df)} teams to {OUTPUT_PATH}")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()