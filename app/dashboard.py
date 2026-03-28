"""
NBA-Simulverse Streamlit Dashboard
Physics-based basketball simulation engine
"""
import sys
import json
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from nba_gpt.config import TARGET_STATS, TRAIN_CONFIG
from nba_gpt.simulation.engine import ScenarioOverride, simulate
from nba_gpt.simulation.roster import RosterScenario, simulate_roster

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAT_LABELS = {
    "points": "PTS",
    "reboundsTotal": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "threePointersMade": "3PM",
}

ENSEMBLE_DIR = TRAIN_CONFIG.checkpoint_dir / "ensemble"
BEST_PT = TRAIN_CONFIG.checkpoint_dir / "best.pt"

SLICED_EVAL_PATH = Path(__file__).parent.parent / "data" / "processed" / "sliced_eval.json"

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model checkpoint…")
def _get_checkpoint_path() -> Path | None:
    """Return ensemble dir sentinel or single best.pt path."""
    has_ensemble = (
        ENSEMBLE_DIR.exists()
        and any(ENSEMBLE_DIR.glob("seed_*/best.pt"))
    )
    if has_ensemble:
        return None  # engine.py auto-detects ensemble when path=None
    if BEST_PT.exists():
        return BEST_PT
    return None  # let engine raise a clear error


@st.cache_resource(show_spinner="Loading sliced evaluation data…")
def _load_sliced_eval() -> dict:
    if not SLICED_EVAL_PATH.exists():
        return {}
    with open(SLICED_EVAL_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _scenario_sliders(key_prefix: str) -> dict:
    """Render the shared scenario sliders; return a dict of values."""
    minutes = st.slider(
        "Minutes", 20, 48, 36,
        key=f"{key_prefix}_minutes",
    )
    rest_days = st.slider(
        "Rest days", 0, 7, 2,
        key=f"{key_prefix}_rest",
    )
    opp_def = st.slider(
        "Opponent pts allowed (lower = tougher defense)", 95, 120, 108,
        key=f"{key_prefix}_opp",
    )
    pace = st.slider(
        "Game pace", 85, 110, 100,
        key=f"{key_prefix}_pace",
    )
    home = st.radio(
        "Location", ["Home", "Away"],
        horizontal=True,
        key=f"{key_prefix}_home",
    )
    return {
        "minutes": float(minutes),
        "rest_days": float(rest_days),
        "opp_pts_allowed": float(opp_def),
        "game_pace": float(pace),
        "home": home == "Home",
    }


def _build_override(params: dict) -> ScenarioOverride:
    return ScenarioOverride(
        minutes=params["minutes"],
        rest_days=params["rest_days"],
        opp_pts_allowed=params["opp_pts_allowed"],
        game_pace=params["game_pace"],
        home=params["home"],
    )


def _result_to_df(result) -> pd.DataFrame:
    """Convert a SimulationResult to a display DataFrame."""
    rows = []
    for stat in TARGET_STATS:
        rows.append({
            "Stat": STAT_LABELS[stat],
            "Mean": round(result.mean[stat], 2),
            "Std": round(result.std[stat], 2),
            "P10": round(result.p10[stat], 2),
            "P25": round(result.p25[stat], 2),
            "P75": round(result.p75[stat], 2),
            "P90": round(result.p90[stat], 2),
        })
    return pd.DataFrame(rows)


def _stat_bar_chart(result, title: str = "") -> go.Figure:
    """Horizontal bar chart of mean ± std for each stat."""
    labels = [STAT_LABELS[s] for s in TARGET_STATS]
    means = [result.mean[s] for s in TARGET_STATS]
    stds = [result.std[s] for s in TARGET_STATS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=means,
        y=labels,
        orientation="h",
        error_x={"type": "data", "array": stds, "visible": True},
        marker_color="#1d78bf",
        name="Projected",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Projected value",
        yaxis={"autorange": "reversed"},
        height=320,
        margin={"t": 40, "b": 20, "l": 60, "r": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _delta_bar_chart(result_a, result_b) -> go.Figure:
    """Grouped bar chart showing both scenarios + delta."""
    labels = [STAT_LABELS[s] for s in TARGET_STATS]
    vals_a = [result_a.mean[s] for s in TARGET_STATS]
    vals_b = [result_b.mean[s] for s in TARGET_STATS]
    deltas = [result_b.mean[s] - result_a.mean[s] for s in TARGET_STATS]
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Scenario A", x=labels, y=vals_a, marker_color="#1d78bf"))
    fig.add_trace(go.Bar(name="Scenario B", x=labels, y=vals_b, marker_color="#e87722"))
    fig.add_trace(go.Bar(
        name="Delta (B − A)",
        x=labels,
        y=deltas,
        marker_color=colors,
        opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title="Scenario comparison — projected stats",
        yaxis_title="Value",
        height=380,
        margin={"t": 50, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Player Simulator
# ---------------------------------------------------------------------------


def tab_player_simulator(checkpoint_path: Path | None) -> None:
    st.subheader("Player Simulator")
    st.write("Simulate a single player's next game under hypothetical conditions.")

    player_name = st.text_input("Player name", value="LeBron James", key="sim_player")

    st.markdown("#### Game conditions")
    params = _scenario_sliders("sim")

    if st.button("Simulate", key="sim_btn", type="primary"):
        if not player_name.strip():
            st.error("Enter a player name.")
            return
        with st.spinner(f"Running simulation for {player_name}…"):
            try:
                override = _build_override(params)
                result = simulate(player_name, override, checkpoint_path=checkpoint_path)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

        st.success(f"Projection for **{result.player_name}**")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.dataframe(_result_to_df(result), use_container_width=True, hide_index=True)
        with col_right:
            st.plotly_chart(
                _stat_bar_chart(result, title=f"{result.player_name} — projected stats"),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 2: Counterfactual Compare
# ---------------------------------------------------------------------------


def tab_counterfactual(checkpoint_path: Path | None) -> None:
    st.subheader("Counterfactual Compare")
    st.write("Compare two hypothetical game scenarios side-by-side.")

    player_name = st.text_input("Player name", value="Stephen Curry", key="cf_player")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Scenario A")
        params_a = _scenario_sliders("cf_a")
    with col_b:
        st.markdown("#### Scenario B")
        params_b = _scenario_sliders("cf_b")

    if st.button("Compare", key="cf_btn", type="primary"):
        if not player_name.strip():
            st.error("Enter a player name.")
            return
        with st.spinner("Running both scenarios…"):
            try:
                result_a = simulate(player_name, _build_override(params_a), checkpoint_path=checkpoint_path)
                result_b = simulate(player_name, _build_override(params_b), checkpoint_path=checkpoint_path)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

        st.success(f"Results for **{result_a.player_name}**")

        # Delta chart
        st.plotly_chart(_delta_bar_chart(result_a, result_b), use_container_width=True)

        # Comparison table
        rows = []
        for stat in TARGET_STATS:
            a_val = result_a.mean[stat]
            b_val = result_b.mean[stat]
            delta = b_val - a_val
            rows.append({
                "Stat": STAT_LABELS[stat],
                "Scenario A": round(a_val, 2),
                "Scenario B": round(b_val, 2),
                "Delta (B − A)": round(delta, 2),
            })
        df = pd.DataFrame(rows)

        def _color_delta(val):
            if val > 0:
                return "color: #2ecc71"
            if val < 0:
                return "color: #e74c3c"
            return ""

        st.dataframe(
            df.style.map(_color_delta, subset=["Delta (B − A)"]),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: Roster Impact
# ---------------------------------------------------------------------------


def tab_roster_impact(checkpoint_path: Path | None) -> None:
    st.subheader("Roster Impact")
    st.write("Simulate a team's box score when a player is absent.")

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team name", value="Warriors", key="roster_team")
    with col2:
        absent_player = st.text_input("Absent player", value="Draymond Green", key="roster_absent")

    home = st.radio("Location", ["Home", "Away"], horizontal=True, key="roster_home")

    if st.button("Simulate Roster", key="roster_btn", type="primary"):
        if not team_name.strip() or not absent_player.strip():
            st.error("Enter both team and absent player.")
            return
        with st.spinner(f"Simulating {team_name} without {absent_player}…"):
            try:
                scenario = RosterScenario(
                    absent_player=absent_player,
                    home=(home == "Home"),
                    n_games=1,
                )
                result = simulate_roster(team_name, scenario, checkpoint_path=checkpoint_path)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

        st.success(f"Roster projection — **{result.team_name}** (without {result.scenario.absent_player})")

        # Build per-player table
        rows = []
        for name, player_result in result.player_results.items():
            row = {"Player": name}
            for stat in TARGET_STATS:
                row[STAT_LABELS[stat]] = round(player_result.mean[stat], 1)
            rows.append(row)

        if not rows:
            st.warning("No player results returned.")
            return

        df = pd.DataFrame(rows)

        # Add team totals row
        totals = {"Player": "TEAM TOTAL"}
        for stat in TARGET_STATS:
            totals[STAT_LABELS[stat]] = round(
                sum(r.mean[stat] for r in result.player_results.values()), 1
            )
        df_totals = pd.DataFrame([totals])
        df_display = pd.concat([df, df_totals], ignore_index=True)

        def _bold_total(row):
            if row["Player"] == "TEAM TOTAL":
                return ["font-weight: bold; background-color: rgba(255,255,255,0.08)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_display.style.apply(_bold_total, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Absent player reference
        if result.absent_baseline:
            st.markdown("---")
            st.markdown(f"**{result.scenario.absent_player}'s baseline projection** (for reference):")
            ref_cols = st.columns(len(TARGET_STATS))
            for col, stat in zip(ref_cols, TARGET_STATS):
                col.metric(STAT_LABELS[stat], f"{result.absent_baseline[stat]:.1f}")


# ---------------------------------------------------------------------------
# Tab 4: Sliced Evaluation
# ---------------------------------------------------------------------------


def tab_sliced_eval() -> None:
    st.subheader("Sliced Evaluation")
    st.write("NBA-GPT vs XGBoost baseline — mean absolute error across data slices.")

    data = _load_sliced_eval()
    if not data:
        st.warning(f"Could not load sliced evaluation data from `{SLICED_EVAL_PATH}`.")
        return

    # Build DataFrame: rows=slices, columns=stats, values=GPT MAE
    slice_names = list(data.keys())
    stat_names = list(STAT_LABELS.keys())

    gpt_rows = []
    xgb_rows = []
    win_rows = []  # True = GPT wins (lower MAE)

    for sl in slice_names:
        gpt_row = {}
        xgb_row = {}
        win_row = {}
        for stat in stat_names:
            g = data[sl]["gpt_mae"].get(stat, float("nan"))
            x = data[sl]["xgb_mae"].get(stat, float("nan"))
            gpt_row[STAT_LABELS[stat]] = round(g, 3)
            xgb_row[STAT_LABELS[stat]] = round(x, 3)
            win_row[STAT_LABELS[stat]] = g < x
        gpt_rows.append(gpt_row)
        xgb_rows.append(xgb_row)
        win_rows.append(win_row)

    slice_labels = [s.replace("_", " ").title() for s in slice_names]
    df_gpt = pd.DataFrame(gpt_rows, index=slice_labels)
    df_xgb = pd.DataFrame(xgb_rows, index=slice_labels)
    df_wins = pd.DataFrame(win_rows, index=slice_labels)

    # Summary metric
    total_cells = df_wins.size
    gpt_wins_count = int(df_wins.values.sum())
    st.metric(
        "NBA-GPT wins",
        f"{gpt_wins_count} / {total_cells} stat-slice combinations",
        delta=f"{gpt_wins_count - (total_cells - gpt_wins_count)} vs XGBoost",
    )

    # Heatmap of GPT MAE
    st.markdown("#### GPT MAE heatmap (green = GPT better, red = GPT worse)")

    # Build z-scores relative to XGB for color coding (negative = GPT better)
    z_values = (df_gpt.values - df_xgb.values)  # negative => GPT wins

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=df_gpt.columns.tolist(),
        y=df_gpt.index.tolist(),
        colorscale=[
            [0.0, "#2ecc71"],   # GPT much better
            [0.5, "#f8f8f8"],   # tied
            [1.0, "#e74c3c"],   # GPT much worse
        ],
        zmid=0,
        text=df_gpt.values.round(3),
        texttemplate="%{text}",
        hovertemplate="Slice: %{y}<br>Stat: %{x}<br>GPT MAE: %{text}<extra></extra>",
        showscale=True,
        colorbar={"title": "GPT MAE − XGB MAE"},
    ))
    fig.update_layout(
        height=max(300, len(slice_names) * 55),
        margin={"t": 20, "b": 60, "l": 140, "r": 20},
        xaxis={"side": "top"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    with st.expander("View full comparison table (GPT vs XGBoost)"):
        view_cols = st.columns(2)
        with view_cols[0]:
            st.markdown("**NBA-GPT MAE**")
            st.dataframe(df_gpt, use_container_width=True)
        with view_cols[1]:
            st.markdown("**XGBoost MAE**")
            st.dataframe(df_xgb, use_container_width=True)

    # Per-slice wins summary
    st.markdown("#### Win count per slice")
    win_counts = df_wins.sum(axis=1).astype(int)
    total_stats = len(stat_names)
    summary_df = pd.DataFrame({
        "Slice": slice_labels,
        "GPT wins": win_counts.values,
        f"/ {total_stats} stats": [f"{v}/{total_stats}" for v in win_counts.values],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="NBA-Simulverse",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Header
    st.title("NBA-Simulverse")
    st.markdown(
        "<p style='font-size:1.1rem; color:#888; margin-top:-12px;'>"
        "Physics-based basketball simulation engine"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    checkpoint_path = _get_checkpoint_path()

    # Ensemble / single-model status badge
    has_ensemble = (
        ENSEMBLE_DIR.exists()
        and any(ENSEMBLE_DIR.glob("seed_*/best.pt"))
    )
    if has_ensemble:
        n_seeds = len(list(ENSEMBLE_DIR.glob("seed_*/best.pt")))
        st.caption(f"Model: ensemble ({n_seeds} seeds) — uncertainty from model disagreement")
    elif BEST_PT.exists():
        st.caption("Model: single checkpoint (MC dropout uncertainty) — run `09_train_ensemble.py` for calibrated uncertainty")
    else:
        st.warning("No model checkpoint found. Run the training pipeline first.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Player Simulator",
        "Counterfactual Compare",
        "Roster Impact",
        "Sliced Evaluation",
    ])

    with tab1:
        tab_player_simulator(checkpoint_path)

    with tab2:
        tab_counterfactual(checkpoint_path)

    with tab3:
        tab_roster_impact(checkpoint_path)

    with tab4:
        tab_sliced_eval()


if __name__ == "__main__":
    main()
