import pandas as pd
from pathlib import Path
from collections import defaultdict

DYNAMIC_DIR = Path("../data/dynamic_events_pl_24/dynamic")
OUTPUT_PATH = Path("../data/team_build_up_analyst/buildup_metrics.csv")

# 10 FPS → 7 seconds = 70 frames
SEVEN_SECONDS_FRAMES = 70

OUT_OF_PLAY_PHASES = {'set_play', 'disruption'}


def time_to_seconds(t):
    """Convert mm:ss.t time format to seconds."""
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
    phase_data = {}
    for phase_idx, grp in df[df['event_type'] == 'player_possession'].groupby('phase_index'):
        phase_data[phase_idx] = {
            'team': grp['team_shortname'].iloc[0],
            'phase_type': grp['team_in_possession_phase_type'].iloc[0],
            'had_turnover': bool(grp['team_possession_loss_in_phase'].any()),
        }

    sorted_indices = sorted(phase_data.keys())
    pos_lookup = {idx: i for i, idx in enumerate(sorted_indices)}

    return phase_data, sorted_indices, pos_lookup


def chain_reaches_finish(start_phase_idx, team, phase_data, sorted_indices, pos_lookup):

    if start_phase_idx not in pos_lookup:
        return False

    if phase_data[start_phase_idx]['had_turnover']:
        return False

    pos = pos_lookup[start_phase_idx] + 1

    while pos < len(sorted_indices):

        idx = sorted_indices[pos]
        info = phase_data[idx]

        if info['team'] != team:
            return False

        if info['phase_type'] in OUT_OF_PLAY_PHASES:
            return False

        if info['phase_type'] == 'finish':
            return True

        if info['had_turnover']:
            return False

        pos += 1

    return False


def compute_match_team_metrics(df, team, phase_data, sorted_indices, pos_lookup):

    pp_bu = df[
        (df['event_type'] == 'player_possession') &
        (df['team_in_possession_phase_type'] == 'build_up') &
        (df['team_shortname'] == team)
    ]

    if pp_bu.empty:
        return None

    phase_summary = (
        pp_bu.groupby('phase_index', sort=True)
        .agg(
            had_turnover=('team_possession_loss_in_phase', 'first'),
            next_phase=('current_team_in_possession_next_phase_type', 'first'),
            last_frame=('frame_start', 'max'),
            last_period=('period', 'last'),

            goalkeeper_involved=('player_in_possession_position', lambda s: (s == 'GK').any()),
            successful_passes=('pass_outcome', lambda s: (s == 'successful').sum()),
            start_time=('time_start', 'first'),
            end_time=('time_end', 'last'),
            players_involved=('player_in_possession_id', lambda s: s.dropna().nunique()),
            last_channel=('channel_start', 'last')
        )
        .reset_index()
    )

    if phase_summary.empty:
        return None

    # Duration
    phase_summary["start_seconds"] = phase_summary["start_time"].apply(time_to_seconds)
    phase_summary["end_seconds"] = phase_summary["end_time"].apply(time_to_seconds)
    phase_summary["duration"] = phase_summary["end_seconds"] - phase_summary["start_seconds"]

    n_phases = len(phase_summary)
    n_turnovers = int((phase_summary['had_turnover'] == True).sum())

    no_to = phase_summary['had_turnover'] == False
    nxt = phase_summary['next_phase']

    n_to_create = int(((nxt == 'create') & no_to).sum())
    n_to_direct = int(((nxt == 'direct') & no_to).sum())

    n_chain_to_finish = sum(
        1 for phase_idx in phase_summary['phase_index']
        if chain_reaches_finish(phase_idx, team, phase_data, sorted_indices, pos_lookup)
    )

    flb_phases = set(pp_bu[pp_bu['first_line_break'] == True]['phase_index'].unique())
    slb_phases = set(pp_bu[pp_bu['second_last_line_break'] == True]['phase_index'].unique())

    n_first_lb = sum(1 for ph in phase_summary['phase_index'] if ph in flb_phases)
    n_second_lb = sum(1 for ph in phase_summary['phase_index'] if ph in slb_phases)

    turnover_phases = phase_summary[phase_summary['had_turnover'] == True]

    total_box_entries = 0
    total_xshot_max = 0.0

    for _, row in turnover_phases.iterrows():

        frame_t = row['last_frame']
        period = row['last_period']

        window = df[
            (df['frame_start'] > frame_t) &
            (df['frame_start'] <= frame_t + SEVEN_SECONDS_FRAMES) &
            (df['period'] == period)
        ]

        box_pp = window[
            (window['event_type'] == 'player_possession') &
            (window['team_shortname'] != team) &
            (window['penalty_area_start'] == True)
        ]

        total_box_entries += len(box_pp)

        obe_press = window[
            (window['event_type'] == 'on_ball_engagement') &
            (window['team_shortname'] == team)
        ]

        xshot_vals = obe_press['xshot_player_possession_max'].dropna()

        total_xshot_max += xshot_vals.max() if len(xshot_vals) > 0 else 0.0

    # NEW METRICS
    sum_passes = phase_summary["successful_passes"].sum()
    sum_duration = phase_summary["duration"].sum()
    sum_players = phase_summary["players_involved"].sum()
    sum_gk = phase_summary["goalkeeper_involved"].sum()

    channel_counts = phase_summary["last_channel"].value_counts()

    channel_dict = {}
    for ch, count in channel_counts.items():
        channel_dict[f"channel_{ch}"] = count

    return {
        'n_phases': n_phases,
        'n_turnovers': n_turnovers,
        'n_to_create': n_to_create,
        'n_to_direct': n_to_direct,
        'n_chain_to_finish': n_chain_to_finish,
        'n_first_lb': n_first_lb,
        'n_second_lb': n_second_lb,
        'box_entries': total_box_entries,
        'xshot_max': total_xshot_max,

        'sum_passes': sum_passes,
        'sum_duration': sum_duration,
        'sum_players': sum_players,
        'sum_gk': sum_gk,

        **channel_dict
    }


def main():

    parquet_files = sorted(DYNAMIC_DIR.glob("*.parquet"))

    print(f"Processing {len(parquet_files)} match files...")

    team_totals = defaultdict(lambda: defaultdict(float))
    team_matches = defaultdict(set)

    for i, f in enumerate(parquet_files, 1):

        if i % 50 == 0:
            print(f"  {i}/{len(parquet_files)}")

        df = pd.read_parquet(f)

        phase_data, sorted_indices, pos_lookup = build_phase_data(df)

        teams = df[df['event_type'] == 'player_possession']['team_shortname'].unique()

        for team in teams:

            result = compute_match_team_metrics(
                df, team, phase_data, sorted_indices, pos_lookup
            )

            if result:

                for k, v in result.items():
                    team_totals[team][k] += v

                team_matches[team].add(f.name)

    rows = []

    for team, t in team_totals.items():

        n = t['n_phases']
        nt = t['n_turnovers']

        row = {

            'team': team,

            'buildup_to_create_pct':
                round(100 * t['n_to_create'] / n, 2) if n > 0 else 0,

            'buildup_to_direct_pct':
                round(100 * t['n_to_direct'] / n, 2) if n > 0 else 0,

            'buildup_that_ends_with_finish_pct':
                round(100 * t['n_chain_to_finish'] / n, 2) if n > 0 else 0,

            'first_line_break_pct_buildup':
                round(100 * t['n_first_lb'] / n, 2) if n > 0 else 0,

            'second_last_line_break_pct_buildup':
                round(100 * t['n_second_lb'] / n, 2) if n > 0 else 0,

            'turnover_pct_buildup':
                round(100 * nt / n, 2) if n > 0 else 0,

            'opp_box_entries_within_7s_after_turnover':
                round(t['box_entries'] / nt, 4) if nt > 0 else 0,

            'opp_shot_probability_within_7s_after_turnover':
                round(t['xshot_max'] / nt, 4) if nt > 0 else 0,

            'prop_gk_involved':
                round(t['sum_gk'] / n, 3) if n > 0 else 0,

            'avg_passes':
                round(t['sum_passes'] / n, 2) if n > 0 else 0,

            'avg_duration':
                round(t['sum_duration'] / n, 2) if n > 0 else 0,

            'avg_players_involved':
                round(t['sum_players'] / n, 2) if n > 0 else 0,

            'build_ups_per_game':
                round(n / len(team_matches[team]), 2)
        }

        # channel proportions
        channel_cols = [k for k in t.keys() if k.startswith("channel_")]

        for ch in channel_cols:
            row[f"prop_{ch}"] = round(t[ch] / n, 3)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values('team').reset_index(drop=True)

    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(out)} teams to {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()