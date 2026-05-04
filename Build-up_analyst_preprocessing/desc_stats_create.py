import pandas as pd
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_input_path(path_value):
    """Resolve a path from cwd first, then script-dir and repo-root fallbacks."""
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = SCRIPT_DIR / candidate
    if script_candidate.exists():
        return script_candidate

    root_candidate = REPO_ROOT / candidate
    if root_candidate.exists():
        return root_candidate

    return candidate


def is_non_zero_value(value):
    """Return True if value is not NA/empty/zero-like."""
    if pd.isna(value):
        return False

    value_str = str(value).strip().lower()
    return value_str not in {"", "0", "0.0", "none", "nan"}


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


def first_non_empty_index(series):
    """Return index of first non-empty value in a pandas Series."""
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    return non_empty.index[0] if len(non_empty) > 0 else None


def normalized_first_non_empty(series):
    """Return first non-empty value as a normalized lowercase string."""
    value = first_non_empty(series)
    if pd.isna(value):
        return None
    value_str = str(value).strip().lower()
    return value_str if value_str != "" else None


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
    folder = resolve_input_path(matchdata_folder)
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
    team_map = pd.read_csv(resolve_input_path(team_map_file))

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


def build_team_carry_stats(matchdata_folder="matchdata"):
    """Count build-up carries per team using the same conditions as the player carry metric."""
    folder = resolve_input_path(matchdata_folder)
    files = sorted(folder.glob("*.parquet"))

    carry_rows = []

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
            pp = df[df["event_type"] == "player_possession"].copy() if "event_type" in df.columns else df.copy()

            if "carry" not in pp.columns:
                continue

            x_delta = pp["x_end"].astype(float) - pp["x_start"].astype(float)
            bu_carries = pp[
                pp["carry"].eq(True) &
                (pp["team_in_possession_phase_type_id"] == 0) &
                pp["player_position"].ne("GK") &
                pp["x_start"].notna() &
                pp["x_end"].notna() &
                (x_delta >= 5.0) &
                pp["separation_start"].notna() &
                (pp["separation_start"].astype(float) <= 10.0)
            ]

            if not bu_carries.empty:
                carry_rows.append(bu_carries[["team_id"]].copy())

        except Exception as e:
            print(f"  carry count error {file_path.name}: {e}")

    if not carry_rows:
        return pd.DataFrame(columns=["team_id", "build_up_carries"])

    return (
        pd.concat(carry_rows, ignore_index=True)
        .groupby("team_id")
        .size()
        .reset_index(name="build_up_carries")
    )


def get_team_label(row):
    """Get a best-effort team label directly from event row columns."""
    for col in ["team_name", "team_shortname", "team"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != "":
            return row[col]
    return pd.NA


def build_player_pass_rec_stats(matchdata_folder="matchdata", team_map_file="team_id_name_map.csv"):
    """Build player-level short and long pass/reception counts from build-up transitions."""
    folder = resolve_input_path(matchdata_folder)
    files = sorted(folder.glob("*.parquet"))

    event_rows = []

    for file_path in files:
        try:
            df = pd.read_parquet(file_path).copy()

            required_cols = {
                "team_in_possession_phase_type_id",
                "current_team_in_possession_next_phase_type",
                "team_possession_loss_in_phase",
                "end_type",
                "pass_outcome",
                "player_name",
            }

            if not required_cols.issubset(df.columns):
                continue

            if "event_type" in df.columns:
                player_df = df[df["event_type"] == "player_possession"].copy()
            else:
                player_df = df.copy()

            if player_df.empty:
                continue

            if "phase_index" in player_df.columns:
                phase_col = "phase_index"
            else:
                # Fallback for datasets without phase_index.
                # Include team changes so phase boundaries do not merge possessions.
                df["phase_id"] = (
                    df["team_in_possession_phase_type_id"]
                    .ne(df["team_in_possession_phase_type_id"].shift())
                    | df["team_id"].ne(df["team_id"].shift())
                ).cumsum()
                player_df["phase_id"] = df.loc[player_df.index, "phase_id"]
                phase_col = "phase_id"

            g = player_df.groupby(phase_col, sort=False)

            cond_build_up = g["team_in_possession_phase_type_id"].first() == 0
            cond_no_turnover = ~g["team_possession_loss_in_phase"].apply(
                lambda s: s.eq(True).any()
            )

            candidate_phase_ids = sorted(
                (cond_build_up & cond_no_turnover)
                .loc[lambda s: s]
                .index
                .tolist()
            )

            ordered_phase_ids = list(g.groups.keys())
            phase_pos_map = {phase_id: idx for idx, phase_id in enumerate(ordered_phase_ids)}

            for phase_id in candidate_phase_ids:
                if phase_id not in phase_pos_map:
                    continue

                build_up_rows = player_df[player_df[phase_col] == phase_id]
                if build_up_rows.empty:
                    continue

                build_up_next_phase_type = normalized_first_non_empty(
                    build_up_rows["current_team_in_possession_next_phase_type"]
                )
                if build_up_next_phase_type not in {"direct", "create"}:
                    continue

                next_pos = phase_pos_map[phase_id] + 1
                if next_pos >= len(ordered_phase_ids):
                    continue

                next_phase_id = ordered_phase_ids[next_pos]
                next_phase_rows = player_df[player_df[phase_col] == next_phase_id]
                if next_phase_rows.empty:
                    continue

                if build_up_next_phase_type == "create":
                    valid_end_rows = build_up_rows[
                        build_up_rows["end_type"].apply(is_non_zero_value)
                    ]
                    if valid_end_rows.empty:
                        continue

                    last_end_row = valid_end_rows.iloc[-1]
                    if str(last_end_row["end_type"]).strip().lower() != "pass":
                        continue

                    if str(last_end_row["pass_outcome"]).strip().lower() != "successful":
                        continue

                    # Forward condition: pass must move toward opponent's goal (x increases)
                    x_start_val = last_end_row.get("x_start", pd.NA)
                    x_end_val   = last_end_row.get("x_end",   pd.NA)
                    if pd.isna(x_start_val) or pd.isna(x_end_val):
                        continue
                    if float(x_end_val) <= float(x_start_val):
                        continue

                    passer_name = last_end_row["player_name"]
                    passer_team_id = last_end_row["team_id"] if "team_id" in last_end_row.index else pd.NA
                    passer_team_label = get_team_label(last_end_row)

                    if "player_targeted_name" not in last_end_row.index:
                        continue

                    receiver_name = last_end_row["player_targeted_name"]
                    if not is_non_zero_value(receiver_name):
                        continue

                    event_rows.append(
                        {
                            "player_name": passer_name,
                            "team_id": passer_team_id,
                            "team_label": passer_team_label,
                            "short_pass": 1,
                            "short_rec": 0,
                            "long_pass": 0,
                            "long_rec": 0,
                            "carry": 0,
                        }
                    )
                    event_rows.append(
                        {
                            "player_name": receiver_name,
                            "team_id": passer_team_id,
                            "team_label": passer_team_label,
                            "short_pass": 0,
                            "short_rec": 1,
                            "long_pass": 0,
                            "long_rec": 0,
                            "carry": 0,
                        }
                    )

                if build_up_next_phase_type == "direct":
                    direct_pass_rows = next_phase_rows[
                        next_phase_rows["end_type"].astype(str).str.strip().str.lower() == "pass"
                    ]
                    if direct_pass_rows.empty:
                        continue

                    last_direct_end_row = direct_pass_rows.iloc[-1]

                    if str(last_direct_end_row["pass_outcome"]).strip().lower() != "successful":
                        continue

                    passer_name = last_direct_end_row["player_name"]
                    passer_team_id = (
                        last_direct_end_row["team_id"] if "team_id" in last_direct_end_row.index else pd.NA
                    )
                    passer_team_label = get_team_label(last_direct_end_row)

                    if "player_targeted_name" not in last_direct_end_row.index:
                        continue

                    receiver_name = last_direct_end_row["player_targeted_name"]

                    if not is_non_zero_value(receiver_name):
                        continue

                    event_rows.append(
                        {
                            "player_name": passer_name,
                            "team_id": passer_team_id,
                            "team_label": passer_team_label,
                            "short_pass": 0,
                            "short_rec": 0,
                            "long_pass": 1,
                            "long_rec": 0,
                            "carry": 0,
                        }
                    )
                    event_rows.append(
                        {
                            "player_name": receiver_name,
                            "team_id": passer_team_id,
                            "team_label": passer_team_label,
                            "short_pass": 0,
                            "short_rec": 0,
                            "long_pass": 0,
                            "long_rec": 1,
                            "carry": 0,
                        }
                    )

                # --- Carry metric ---
                # Build-up carry under pressure with significant forward progress:
                # - carry == True (player covered ≥ 2m)
                # - player_position != "GK" (exclude goalkeeper)
                # - x_end - x_start >= 5.0 (≥ 5m forward; coords in metres)
                # - separation_start <= 10.0 (nearest opponent ≤ 10m at carry start)
                if "carry" not in build_up_rows.columns:
                    carry_rows = pd.DataFrame()
                else:
                    x_delta = (
                        build_up_rows["x_end"].astype(float) -
                        build_up_rows["x_start"].astype(float)
                    )
                    carry_rows = build_up_rows[
                        build_up_rows["carry"].eq(True) &
                        build_up_rows["player_position"].ne("GK") &
                        build_up_rows["x_start"].notna() &
                        build_up_rows["x_end"].notna() &
                        (x_delta >= 5.0) &
                        build_up_rows["separation_start"].notna() &
                        (build_up_rows["separation_start"].astype(float) <= 10.0)
                    ]

                for _, carry_row in carry_rows.iterrows():
                    event_rows.append(
                        {
                            "player_name": carry_row["player_name"],
                            "team_id": carry_row.get("team_id", pd.NA),
                            "team_label": get_team_label(carry_row),
                            "short_pass": 0,
                            "short_rec": 0,
                            "long_pass": 0,
                            "long_rec": 0,
                            "carry": 1,
                        }
                    )

        except Exception as e:
            print(f"Error processing {file_path.name} for player stats: {e}")

    if not event_rows:
        return pd.DataFrame(
            columns=[
                "Player Name",
                "Team Name",
                "short_pass",
                "short_rec",
                "long_pass",
                "long_rec",
                "carry",
            ]
        )

    player_events = pd.DataFrame(event_rows)

    if "team_id" in player_events.columns:
        team_map = pd.read_csv(resolve_input_path(team_map_file))
        if {"team_id", "team_name"}.issubset(team_map.columns):
            team_lookup = team_map[["team_id", "team_name"]].drop_duplicates(subset=["team_id"])
            player_events = player_events.merge(team_lookup, on="team_id", how="left")
            player_events["Team Name"] = player_events["team_name"].where(
                player_events["team_name"].notna(),
                player_events["team_label"],
            )
        else:
            player_events["Team Name"] = player_events["team_label"]
    else:
        player_events["Team Name"] = player_events["team_label"]

    player_events["Team Name"] = player_events["Team Name"].fillna("Unknown")

    final_player_stats = (
        player_events.groupby(["player_name", "Team Name"], dropna=False)[
            ["short_pass", "short_rec", "long_pass", "long_rec", "carry"]
        ]
        .sum()
        .reset_index()
        .rename(columns={"player_name": "Player Name"})
    )

    final_player_stats = final_player_stats[
        ["Player Name", "Team Name", "short_pass", "short_rec", "long_pass", "long_rec", "carry"]
    ].copy()
    final_player_stats = final_player_stats.sort_values(
        by=["short_pass", "short_rec", "long_pass", "long_rec", "carry", "Player Name"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    return final_player_stats


def main():
    matchdata_folder = "data/dynamic_events_pl_24/dynamic"
    team_map_file = "team_id_name_map.csv"
    output_file = str(REPO_ROOT / "data" / "team_build_up_analyst" / "buildup_metrics_style.csv")
    player_output_file = str(REPO_ROOT / "data" / "team_build_up_analyst" / "player_pass_rec_stats.csv")

    final_clean = build_team_stats(
        matchdata_folder=matchdata_folder,
        team_map_file=team_map_file,
    )

    if final_clean.empty:
        print("No final results produced.")
        return

    # Add team-level build_up_carries
    carry_stats = build_team_carry_stats(matchdata_folder=matchdata_folder)
    if not carry_stats.empty:
        team_map_df = pd.read_csv(resolve_input_path(team_map_file))
        carry_named = carry_stats.merge(team_map_df[["team_id", "team_name"]], on="team_id", how="left")
        final_clean = final_clean.merge(
            carry_named[["team_name", "build_up_carries"]], on="team_name", how="left"
        )
        final_clean["build_up_carries"] = final_clean["build_up_carries"].fillna(0).astype(int)
    else:
        final_clean["build_up_carries"] = 0

    print("\nFinal team statistics:")
    print(final_clean)

    final_clean.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved results to: {output_file}")

    final_player_stats = build_player_pass_rec_stats(
        matchdata_folder=matchdata_folder,
        team_map_file=team_map_file,
    )

    if final_player_stats.empty:
        print("\nNo player pass/reception results produced.")
        return

    print("\nFinal player pass/reception statistics:")
    print(final_player_stats)

    final_player_stats.to_csv(player_output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved player results to: {player_output_file}")

    print("\n" + "=" * 60)
    print("Top 3 players per team — carry")
    print("=" * 60)
    top_carry = (
        final_player_stats[final_player_stats["carry"] > 0]
        .sort_values(["Team Name", "carry"], ascending=[True, False])
        .groupby("Team Name")
        .head(3)
    )
    for team, grp in top_carry.groupby("Team Name"):
        print(f"\n  {team}")
        for _, row in grp.iterrows():
            print(f"    {row['Player Name']:<30} {int(row['carry'])} carries")


if __name__ == "__main__":
    main()



