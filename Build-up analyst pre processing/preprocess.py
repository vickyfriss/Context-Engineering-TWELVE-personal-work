import pandas as pd
from pathlib import Path
from collections import defaultdict

DYNAMIC_DIR = Path("../data/dynamic_events_pl_24/dynamic")
OUTPUT_PATH = Path("../data/team_build_up_analyst/buildup_metrics.csv")

# 10 FPS → 7 seconds = 70 frames
SEVEN_SECONDS_FRAMES = 70

# Phase types that interrupt a possession chain (ball out of play)
OUT_OF_PLAY_PHASES = {'set_play', 'disruption'}


def build_phase_data(df):
    """
    Build a per-phase lookup dict from PP events.
    Every phase has at least one PP event, so no phase is missed.
    """
    phase_data = {}
    for phase_idx, grp in df[df['event_type'] == 'player_possession'].groupby('phase_index'):
        phase_data[phase_idx] = {
            'team':         grp['team_shortname'].iloc[0],
            'phase_type':   grp['team_in_possession_phase_type'].iloc[0],
            'had_turnover': bool(grp['team_possession_loss_in_phase'].any()),
        }
    sorted_indices = sorted(phase_data.keys())
    pos_lookup = {idx: i for i, idx in enumerate(sorted_indices)}
    return phase_data, sorted_indices, pos_lookup


def chain_reaches_finish(start_phase_idx, team, phase_data, sorted_indices, pos_lookup):
    """
    Returns True if the possession chain from start_phase_idx eventually reaches
    a finish phase for `team`, with no ball loss and no out-of-play interruption.

    Conditions:
    - The start (build_up) phase must have no turnover.
    - Each subsequent phase must belong to the same team.
    - No set_play or disruption phase may appear in between.
    - Reaching a finish phase returns True (regardless of what happens in finish).
    - An intermediate phase with a turnover breaks the chain.
    """
    if start_phase_idx not in pos_lookup:
        return False
    if phase_data[start_phase_idx]['had_turnover']:
        return False

    pos = pos_lookup[start_phase_idx] + 1
    while pos < len(sorted_indices):
        idx  = sorted_indices[pos]
        info = phase_data[idx]

        if info['team'] != team:                        # possession changed to opponent
            return False
        if info['phase_type'] in OUT_OF_PLAY_PHASES:   # ball out of play
            return False
        if info['phase_type'] == 'finish':             # reached finish!
            return True
        if info['had_turnover']:                        # lost ball in intermediate phase
            return False
        pos += 1

    return False


def compute_match_team_metrics(df, team, phase_data, sorted_indices, pos_lookup):
    """Return dict of raw counts/sums for one team from one match's events."""
    pp_bu = df[
        (df['event_type'] == 'player_possession') &
        (df['team_in_possession_phase_type'] == 'build_up') &
        (df['team_shortname'] == team)
    ]
    if pp_bu.empty:
        return None

    # One row per build-up phase.
    # NOTE: last_player_possession_in_team_possession marks the end of an entire
    # possession *sequence* (spanning multiple phases), not the end of a single phase.
    # Using groupby ensures every build-up phase is counted, including those that are
    # intermediate steps in a longer possession sequence.
    # team_possession_loss_in_phase and current_team_in_possession_next_phase_type
    # are identical for all PP events within a phase, so first() is correct.
    phase_summary = (
        pp_bu.groupby('phase_index', sort=True)
        .agg(
            had_turnover=('team_possession_loss_in_phase', 'first'),
            next_phase   =('current_team_in_possession_next_phase_type', 'first'),
            last_frame   =('frame_start', 'max'),
            last_period  =('period', 'last'),
        )
        .reset_index()
    )
    if phase_summary.empty:
        return None

    n_phases    = len(phase_summary)
    n_turnovers = int((phase_summary['had_turnover'] == True).sum())

    no_to = phase_summary['had_turnover'] == False
    nxt   = phase_summary['next_phase']

    n_to_create = int(((nxt == 'create') & no_to).sum())
    n_to_direct = int(((nxt == 'direct') & no_to).sum())

    # Build-up phases whose full possession chain eventually reaches a finish phase
    n_chain_to_finish = sum(
        1 for phase_idx in phase_summary['phase_index']
        if chain_reaches_finish(phase_idx, team, phase_data, sorted_indices, pos_lookup)
    )

    # Line break metrics: phases where any PP event broke the first / second-last line
    # Using sets for O(1) lookup per phase
    flb_phases = set(pp_bu[pp_bu['first_line_break'] == True]['phase_index'].unique())
    slb_phases = set(pp_bu[pp_bu['second_last_line_break'] == True]['phase_index'].unique())
    n_first_lb  = sum(1 for ph in phase_summary['phase_index'] if ph in flb_phases)
    n_second_lb = sum(1 for ph in phase_summary['phase_index'] if ph in slb_phases)

    # Opponent consequences after each build-up turnover
    turnover_phases        = phase_summary[phase_summary['had_turnover'] == True]
    total_box_entries      = 0
    total_xshot_max        = 0.0   # sum of per-turnover peak xshot (averaged later)

    for _, row in turnover_phases.iterrows():
        frame_t = row['last_frame']
        period  = row['last_period']

        window = df[
            (df['frame_start'] > frame_t) &
            (df['frame_start'] <= frame_t + SEVEN_SECONDS_FRAMES) &
            (df['period'] == period)
        ]

        # Opponent PP events entering the penalty area
        box_pp = window[
            (window['event_type'] == 'player_possession') &
            (window['team_shortname'] != team) &
            (window['penalty_area_start'] == True)
        ]
        total_box_entries += len(box_pp)

        # Peak xshot from OBE events by the pressing team (original team now defending).
        # xshot_player_possession_max on OBE = xG of the opponent being pressed.
        # Using max to capture the single most dangerous moment within 7 seconds.
        obe_press = window[
            (window['event_type'] == 'on_ball_engagement') &
            (window['team_shortname'] == team)
        ]
        xshot_vals = obe_press['xshot_player_possession_max'].dropna()
        total_xshot_max += xshot_vals.max() if len(xshot_vals) > 0 else 0.0

    return {
        'n_phases':          n_phases,
        'n_turnovers':       n_turnovers,
        'n_to_create':       n_to_create,
        'n_to_direct':       n_to_direct,
        'n_chain_to_finish': n_chain_to_finish,
        'n_first_lb':        n_first_lb,
        'n_second_lb':       n_second_lb,
        'box_entries':       total_box_entries,
        'xshot_max':         total_xshot_max,
    }


def main():
    parquet_files = sorted(DYNAMIC_DIR.glob("*.parquet"))
    print(f"Processing {len(parquet_files)} match files...")

    team_totals = defaultdict(lambda: defaultdict(float))

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

    rows = []
    for team, t in team_totals.items():
        n  = t['n_phases']
        nt = t['n_turnovers']
        rows.append({
            'team':                                        team,
            'buildup_to_create_pct':                       round(100 * t['n_to_create'] / n, 2)       if n  > 0 else 0,
            'buildup_to_direct_pct':                       round(100 * t['n_to_direct'] / n, 2)       if n  > 0 else 0,
            'buildup_that_ends_with_finish_pct':           round(100 * t['n_chain_to_finish'] / n, 2) if n  > 0 else 0,
            'first_line_break_pct_buildup':                round(100 * t['n_first_lb'] / n, 2)        if n  > 0 else 0,
            'second_last_line_break_pct_buildup':          round(100 * t['n_second_lb'] / n, 2)       if n  > 0 else 0,
            'turnover_pct_buildup':                        round(100 * nt / n, 2)                      if n  > 0 else 0,
            'opp_box_entries_within_7s_after_turnover':    round(t['box_entries'] / nt, 4)             if nt > 0 else 0,
            'opp_shot_probability_within_7s_after_turnover': round(t['xshot_max'] / nt, 4)            if nt > 0 else 0,
        })

    out = pd.DataFrame(rows).sort_values('team').reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} teams to {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
