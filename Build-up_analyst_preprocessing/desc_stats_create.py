import pandas as pd
from pathlib import Path


def time_to_seconds(t):
    """Convert time in format mm:ss.t to seconds."""
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


def first_non_empty(series):
    """Return the first non-empty value in a pandas Series."""
    s = series.dropna()
    s = s[s.astype(str).str.strip() != ""]
    return s.iloc[0] if len(s) > 0 else pd.NA


def process_match_file(file_path):
    """Process one parquet match file and return phase-level summary."""
    df = pd.read_parquet(file_path).copy()

    # Create phase ids based on consecutive equal values
    df["phase_id"] = (
        df["team_in_possession_phase_type_id"]
        .ne(df["team_in_possession_phase_type_id"].shift())
        .cumsum()
    )

    # Phase filtering conditions
    g = df.groupby("phase_id")

    cond1 = g["team_in_possession_phase_type_id"].first() == 0
    cond2 = g["current_team_in_possession_next_phase_type"].apply(
        lambda s: s.isin(["create", "direct"]).any()
    )
    cond3 = ~g["team_possession_loss_in_phase"].apply(
        lambda s: s.eq(True).any()
    )

    phase_keep_mask = cond1 & cond2 & cond3
    df_kept = df[df["phase_id"].map(phase_keep_mask)].copy()

    if df_kept.empty:
        return pd.DataFrame()

    # Majority team_id within phase
    majority_map = df_kept.groupby("phase_id")["team_id"].agg(
        lambda s: s.mode().iloc[0]
    )
    df_kept["team_id_in_poss"] = df_kept["phase_id"].map(majority_map)

    # Phase-level summary
    phase_summary = (
        df_kept.groupby("phase_id")
        .agg(
            team_id_in_poss=("team_id_in_poss", "first"),
            next_phase_type=(
                "current_team_in_possession_next_phase_type",
                first_non_empty,
            ),
            goalkeeper_involved=(
                "player_in_possession_position",
                lambda s: s.eq("GK").any(),
            ),
            successful_passes=("pass_outcome", lambda s: s.eq("successful").sum()),
            start_time=("time_start", "first"),
            end_time=("time_end", "last"),
            n_players_involved=(
                "player_in_possession_id",
                lambda s: s.dropna().nunique(),
            ),
        )
        .reset_index()
    )

    # Duration
    phase_summary["start_seconds"] = phase_summary["start_time"].apply(time_to_seconds)
    phase_summary["end_seconds"] = phase_summary["end_time"].apply(time_to_seconds)
    phase_summary["phase_duration_seconds"] = (
        phase_summary["end_seconds"] - phase_summary["start_seconds"]
    )

    # Exclude zero-duration phases from average
    phase_summary.loc[
        phase_summary["phase_duration_seconds"] == 0,
        "phase_duration_seconds",
    ] = pd.NA

    # Channel from the last player_possession row in the phase
    player_poss = df_kept[df_kept["event_type"] == "player_possession"].copy()

    if not player_poss.empty:
        phase_channels = (
            player_poss.groupby("phase_id")
            .agg(last_channel=("channel_start", "last"))
            .reset_index()
        )
        phase_summary = phase_summary.merge(phase_channels, on="phase_id", how="left")
    else:
        phase_summary["last_channel"] = pd.NA

    phase_summary["match_file"] = file_path.name

    return phase_summary


def build_team_stats(matchdata_folder="matchdata", team_map_file="team_id_name_map.csv"):
    """Build final team-level statistics from all parquet match files."""
    folder = Path(matchdata_folder)
    files = sorted(folder.glob("*.parquet"))

    all_phase_summaries = []

    for file_path in files:
        try:
            phase_summary = process_match_file(file_path)
            if not phase_summary.empty:
                all_phase_summaries.append(phase_summary)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not all_phase_summaries:
        print("No valid phase summaries were created.")
        return pd.DataFrame()

    all_phases = pd.concat(all_phase_summaries, ignore_index=True)

    print(f"Processed {len(files)} files")
    print(f"Total kept build-up phases: {len(all_phases)}")

    # Proportion direct
    phase_types = all_phases[
        all_phases["next_phase_type"].isin(["create", "direct"])
    ].copy()

    counts = (
        phase_types.groupby(["team_id_in_poss", "next_phase_type"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["create", "direct"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[["create", "direct"]]

    proportions = counts.div(counts.sum(axis=1), axis=0).rename(
        columns={"direct": "prop_direct"}
    ).reset_index()[["team_id_in_poss", "prop_direct"]]

    # Team-level core stats
    team_phase_stats = (
        all_phases.groupby("team_id_in_poss")
        .agg(
            prop_gk_involved=("goalkeeper_involved", "mean"),
            avg_passes=("successful_passes", "mean"),
            avg_duration=("phase_duration_seconds", "mean"),
            avg_players_involved=("n_players_involved", "mean"),
            n_phases=("phase_id", "count"),
            n_matches=("match_file", "nunique"),
        )
        .reset_index()
    )

    final_team_stats = proportions.merge(
        team_phase_stats, on="team_id_in_poss", how="outer"
    )

    # Build-ups per game
    final_team_stats["build_ups_per_game"] = (
        final_team_stats["n_phases"] / final_team_stats["n_matches"]
    )

    # Channel proportions
    channel_data = all_phases.dropna(subset=["last_channel"]).copy()

    if not channel_data.empty:
        channel_counts = (
            channel_data.groupby(["team_id_in_poss", "last_channel"])
            .size()
            .unstack(fill_value=0)
        )

        channel_props = channel_counts.div(
            channel_counts.sum(axis=1), axis=0
        ).reset_index()

        channel_props = channel_props.rename(
            columns={
                col: f"prop_channel_{col}"
                for col in channel_props.columns
                if col != "team_id_in_poss"
            }
        )

        final_team_stats = final_team_stats.merge(
            channel_props, on="team_id_in_poss", how="left"
        )

    # Add team names
    team_map = pd.read_csv(team_map_file)

    final_team_stats_named = final_team_stats.merge(
        team_map,
        left_on="team_id_in_poss",
        right_on="team_id",
        how="left",
    )

    # Final columns
    channel_cols = sorted(
        [c for c in final_team_stats_named.columns if str(c).startswith("prop_channel_")]
    )

    final_clean = final_team_stats_named[
        [
            "team_name",
            "prop_direct",
            "prop_gk_involved",
            "avg_passes",
            "avg_duration",
            "avg_players_involved",
            "build_ups_per_game",
        ] + channel_cols
    ].copy()

    # Rounding
    round_dict = {
        "prop_direct": 3,
        "prop_gk_involved": 3,
        "avg_passes": 2,
        "avg_duration": 2,
        "avg_players_involved": 2,
        "build_ups_per_game": 2,
    }
    round_dict.update({col: 3 for col in channel_cols})

    final_clean = final_clean.round(round_dict)

    return final_clean


def main():
    matchdata_folder = "matchdata"
    team_map_file = "team_id_name_map.csv"
    output_file = "team_build_up_stats.csv"

    final_clean = build_team_stats(
        matchdata_folder=matchdata_folder,
        team_map_file=team_map_file,
    )

    if final_clean.empty:
        print("No final results produced.")
        return

    print("\nFinal team statistics:")
    print(final_clean)

    final_clean.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()