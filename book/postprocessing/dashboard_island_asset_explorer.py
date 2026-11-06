"""Interactive, self-contained dashboard for exploring per-island asset-type
composition and operational status at every simulation timestep.

Companion to ``produce_figures_group_equity.py`` (Figures 2-4) and
``flood_overlay_hour48.py`` (direct-flood-only view). Answers questions
like: "at timestep 48, there are 100 islands; the two largest hold 90% of
hospitals, and half of those are not operational" -- i.e. combines the
*road-network-severance* and *direct-cause* channels of the open "decompose
loss of access" task (see CLAUDE.md) into one explorable view.

Data sources
------------
- **Spatial trace** (required for the main panel): a per-asset, per-timestep
  GeoParquet produced by ``book/interdependent_ci_societal_access.py``'s
  ``--spatial-trace-policies`` flag (see ``run_spatial_trace``). Captures
  ``island_id``, ``operational``, ``blocking_reason`` (0=operational,
  1=physically_damaged, 2=hazard_blocked, 3=dependency_blocked,
  4=other_blocked -- see ``src.adaptation.BLOCKING_REASON_LABELS``) and
  ``type`` for every asset at every timestep. ``island_id`` is assigned by
  the same generic road-based mechanism for *every* asset type (including
  electricity substations and, if present, schools) -- see CLAUDE.md
  "EMS/Hospital island bug" root-causing notes -- so the per-island
  breakdown below is valid for all asset types uniformly, even though
  electricity/school *population access* is itself computed via Voronoi
  service areas rather than island reachability (see
  ``config.py::service_area_function_provider_types``). That distinction is
  surfaced as an annotation, not a different grouping mechanism.
- **EMA results archive** (optional context panel): if the Figure 2/3/4
  summary-stats CSVs already produced by ``produce_figures_group_equity.py``
  exist in ``FIG_DIR``, they are shown as read-only tables alongside the new
  panel. If they don't exist, that tab shows a placeholder instead of
  failing -- this dashboard must still work against an archive from a run
  that never produced them.

Island IDs are **not stable across timesteps** (connected-components
labelling is recomputed every timestep as the road network floods/recovers
-- see CLAUDE.md "Previous/present island state"), so islands are ranked
and labelled by size *within each timestep* ("largest island", "2nd
largest", ...), never tracked by raw ``island_id`` across frames. The raw ID
is kept only in hover text for cross-referencing with other exports (e.g.
``asset_road_match_<execution_id>.parquet``).

Output: a single ``dashboard_island_asset_explorer.html`` file with:
(1) a cheap "uncertainty across draws" panel -- % operational over time per
asset type, one line per captured scenario, colored by policy, scales to
many draws since it's just simple time-series lines; and (2) an expensive
per-island, per-timestep deep dive (Plotly frames + slider) for a small,
curated subset of those draws (see ``--detail-policies``/
``--max-detail-runs``), selectable via dropdown. No server process, no new
heavy dependency (Dash/Streamlit).

Usage:
    python book/postprocessing/dashboard_island_asset_explorer.py
    python book/postprocessing/dashboard_island_asset_explorer.py --policies population_impacts_islands_policy,baseline_no_prioritisation
    python book/postprocessing/dashboard_island_asset_explorer.py --detail-policies population_impacts_islands_policy
    python book/postprocessing/dashboard_island_asset_explorer.py \\
        --trace-path "...\\spatial_traces\\spatial_trace_..._scenario0.parquet"

By default all spatial-trace parquets found under --trace-dir are loaded
and contribute to the uncertainty-across-draws panel; only up to
--max-detail-runs of them (preferring DEFAULT_POLICY) get the full,
expensive per-island breakdown. Use --policies to restrict/order which
runs are loaded at all, and --detail-policies to control which subset gets
the expensive panel.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from book.interdependent_ci_societal_access import AMENITY_CATEGORY_LABELS  # noqa: E402

FIG_DIR = REPO_ROOT / "data" / "output" / "paper_figures_group_equity"
DEFAULT_TRACE_GLOB_DIR = (
    REPO_ROOT / "data" / "output" / "output_hazard_maps_ZH_Delfland" / "ema_exports" / "spatial_traces"
)
DEFAULT_POLICY = "population_impacts_islands_policy"
DEFAULT_TOP_N_ISLANDS = 6
DEFAULT_MAX_DETAIL_RUNS = 5

ASSET_TYPE_LABELS = {"msls": "Electricity substation", **AMENITY_CATEGORY_LABELS}
BLOCKING_REASON_LABELS = {
    0: "Operational",
    1: "Physically damaged",
    2: "Hazard-blocked",
    3: "Dependency-blocked",
    4: "Other blocked",
}
BLOCKING_REASON_COLORS = {
    "Operational": "#55A868",
    "Physically damaged": "#C44E52",
    "Hazard-blocked": "#DD8452",
    "Dependency-blocked": "#8172B2",
    "Other blocked": "#937860",
}
# Voronoi/weighted-Voronoi functions resolve population access via service
# areas, not island reachability -- annotated for the user, not used to
# change how the island breakdown itself is grouped (see module docstring).
try:
    from config import get_config  # noqa: E402

    _SERVICE_AREA_TYPES = get_config().get("service_area_function_provider_types", {})
    _VORONOI_METHOD_BY_FUNCTION: Dict[str, str] = {
        func: spec.get("method", "unknown") for func, spec in _SERVICE_AREA_TYPES.items()
    }
except Exception:
    _VORONOI_METHOD_BY_FUNCTION = {"electricity": "simple_voronoi"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=Path, default=None,
                         help="Explicit path to a single spatial-trace parquet, bypassing "
                              "discovery entirely (produces a dashboard with just that one run "
                              "and no run selector).")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_GLOB_DIR,
                         help="Directory to search for spatial-trace parquet files.")
    parser.add_argument("--execution-id", type=str, default=None,
                         help="Restrict discovery to spatial-trace parquets from this single "
                              "execution_id (e.g. '1793801417', visible in the run's own log "
                              "output, such as 'asset_road_match_<execution_id>.parquet'). "
                              "--trace-dir commonly accumulates files from multiple unrelated "
                              "runs; without this filter all of them are mixed into one "
                              "dashboard. Ignored when --trace-path is given.")
    parser.add_argument("--policies", type=str, default=None, metavar="NAME1,NAME2,...",
                         help="Comma-separated policy names to include as selectable runs. "
                              "Default: every 'spatial_trace_*.parquet' found under --trace-dir "
                              f"(with '{DEFAULT_POLICY}' shown first if present).")
    parser.add_argument("--detail-policies", type=str, default=None, metavar="NAME1,NAME2,...",
                         help="Comma-separated policy names to render the full, expensive "
                              "per-island/per-timestep breakdown for (selectable via the run "
                              "dropdown). All discovered runs still contribute to the cheap "
                              f"'uncertainty across draws' panel. Default: up to "
                              f"{DEFAULT_MAX_DETAIL_RUNS} runs, preferring '{DEFAULT_POLICY}'.")
    parser.add_argument("--max-detail-runs", type=int, default=DEFAULT_MAX_DETAIL_RUNS,
                         help="Cap on how many runs get the full per-island breakdown when "
                              f"--detail-policies is not given. Default: {DEFAULT_MAX_DETAIL_RUNS}.")
    parser.add_argument("--top-n-islands", type=int, default=DEFAULT_TOP_N_ISLANDS,
                         help="Number of largest islands (per asset type, per timestep) shown "
                              "individually before collapsing the rest into 'Other islands'. "
                              f"Default: {DEFAULT_TOP_N_ISLANDS}.")
    parser.add_argument("--output-path", type=Path, default=FIG_DIR / "dashboard_island_asset_explorer.html",
                         help="Output HTML path.")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Loading / run discovery
# --------------------------------------------------------------------------
_TRACE_FILENAME_RE = re.compile(r"^spatial_trace_(?P<execution_id>\d+)_(?P<policy>.+)_scenario(?P<scenario>[^.]+)\.parquet$")


class TraceRun:
    """One discoverable (execution_id, policy, scenario) spatial-trace file,
    selectable in the dashboard's run dropdown."""

    def __init__(self, path: Path, execution_id: str, policy: str, scenario: str):
        self.path = path
        self.execution_id = execution_id
        self.policy = policy
        self.scenario = scenario
        self.uncertainties: Dict = load_uncertainties(path)

    @property
    def label(self) -> str:
        return f"{self.policy} (scenario {self.scenario}, run {self.execution_id})"

    @property
    def hover_text(self) -> str:
        """Short, human-readable summary of the sampled uncertainty values
        for this draw, e.g. for Plotly hover tooltips -- answers "what draw
        am I looking at?" beyond the arbitrary scenario index."""
        if not self.uncertainties:
            return f"{self.label}<br><em>no uncertainty metadata (older capture)</em>"
        params = "<br>".join(f"{k}: {v}" for k, v in sorted(self.uncertainties.items()))
        return f"{self.label}<br>{params}"


def load_uncertainties(trace_path: Path) -> Dict:
    """Load the companion '<trace>_uncertainties.json' written alongside the
    parquet (sampled repair-crew counts, fragility steepness, etc. for that
    scenario). Missing/unreadable (e.g. traces captured before this was
    added) degrades to an empty dict rather than failing."""
    companion_path = trace_path.with_name(trace_path.stem + "_uncertainties.json")
    if not companion_path.exists():
        return {}
    try:
        import json
        return json.loads(companion_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_traces(
    trace_dir: Path,
    policies: Optional[List[str]] = None,
    execution_id: Optional[str] = None,
) -> List[TraceRun]:
    """Find every spatial-trace parquet under ``trace_dir``, optionally
    filtered to ``policies`` and/or a single ``execution_id``. Ordered
    newest-first within each policy; ``DEFAULT_POLICY`` is surfaced first if
    present and not otherwise constrained by an explicit ``policies`` order.

    ``trace_dir`` commonly accumulates parquets from several unrelated runs
    (smoke tests, earlier trace passes, the "final" run, ...) since nothing
    ever cleans it up automatically. Without ``execution_id``, runs from all
    of them are silently mixed together into one dashboard -- pass the
    execution_id you actually want (visible in the run's own log output,
    e.g. "asset_road_match_<execution_id>.parquet") to restrict to one."""
    if not trace_dir.exists():
        return []
    runs = []
    for path in trace_dir.glob("spatial_trace_*.parquet"):
        m = _TRACE_FILENAME_RE.match(path.name)
        if not m:
            continue
        if execution_id is not None and m.group("execution_id") != execution_id:
            continue
        runs.append(TraceRun(path, m.group("execution_id"), m.group("policy"), m.group("scenario")))

    if policies:
        wanted = [p.strip() for p in policies if p.strip()]
        runs = [r for r in runs if r.policy in wanted]
        order = {p: i for i, p in enumerate(wanted)}
        runs.sort(key=lambda r: (order.get(r.policy, len(order)), -r.path.stat().st_mtime))
    else:
        runs.sort(key=lambda r: (r.policy != DEFAULT_POLICY, r.policy, -r.path.stat().st_mtime))
    return runs


def load_trace(trace_path: Path) -> pd.DataFrame:
    """Load the spatial-trace parquet, dropping geometry (not needed for the
    per-island count panels, keeps the in-memory frame small)."""
    import geopandas as gpd

    trace_gdf = gpd.read_parquet(trace_path)
    df = pd.DataFrame(trace_gdf.drop(columns="geometry", errors="ignore"))
    missing = {"timestep", "asset_id", "type", "island_id", "operational"} - set(df.columns)
    if missing:
        raise ValueError(f"Spatial trace at {trace_path} is missing required column(s): {missing}")
    if "blocking_reason" not in df.columns:
        # Older traces captured before blocking_reason was added to
        # DEFAULT_SPATIAL_TRACE_FIELDS -- fall back to a coarse 2-way split
        # (operational vs a single generic "blocked" bucket) rather than
        # failing, so this dashboard still works against them.
        df["blocking_reason"] = np.where(df["operational"].astype(bool), 0, 4)
    df["blocking_label"] = df["blocking_reason"].map(BLOCKING_REASON_LABELS).fillna("Other blocked")
    return df


# --------------------------------------------------------------------------
# Aggregation: rank islands by size *within each timestep*, per asset type
# --------------------------------------------------------------------------
def build_ranked_island_frame(trace_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """For every (timestep, asset type), rank islands by how many assets of
    that type they hold (descending), keep the top ``top_n`` individually
    and collapse the remainder into a synthetic 'Other islands' bucket.

    Returns a long frame with one row per (timestep, type, rank_label,
    blocking_label): columns ``timestep, type, rank_label, island_id,
    blocking_label, count``. ``island_id`` is NaN for the 'Other islands'
    bucket (it aggregates many islands) and is otherwise kept only for
    hover-text cross-referencing (not a stable identity across frames).
    """
    rows = []
    group_cols = ["timestep", "type", "island_id"]
    # dropna=False is essential: assets with no island match (e.g. boundary
    # assets that never matched the road network -- see CLAUDE.md "722
    # boundary assets" notes) have island_id == NaN. Pandas groupby drops
    # NaN keys by default, which silently excluded these assets from every
    # bar entirely (an asset that "vanishes" rather than showing up in any
    # blocking-reason colour) -- they are now routed to their own explicit
    # "Unassigned" bucket below instead.
    island_totals = (
        trace_df.groupby(group_cols, observed=True, dropna=False)
        .size()
        .rename("island_total")
        .reset_index()
    )
    blocking_counts = (
        trace_df.groupby(group_cols + ["blocking_label"], observed=True, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    for (timestep, asset_type), sub_totals in island_totals.groupby(["timestep", "type"], observed=True):
        unassigned_mask = sub_totals["island_id"].isna()
        ranked = sub_totals[~unassigned_mask].sort_values("island_total", ascending=False).reset_index(drop=True)
        top = ranked.iloc[:top_n]
        rest_island_ids = set(ranked.iloc[top_n:]["island_id"])

        sub_blocking = blocking_counts[
            (blocking_counts["timestep"] == timestep) & (blocking_counts["type"] == asset_type)
        ]

        for rank, row in top.iterrows():
            island_rows = sub_blocking[sub_blocking["island_id"] == row["island_id"]]
            label = f"Largest island" if rank == 0 else f"{rank + 1}{_ordinal_suffix(rank + 1)} largest"
            for _, brow in island_rows.iterrows():
                rows.append({
                    "timestep": timestep, "type": asset_type, "rank_label": label,
                    "island_id": row["island_id"], "blocking_label": brow["blocking_label"],
                    "count": brow["count"],
                })
            if island_rows.empty:
                rows.append({
                    "timestep": timestep, "type": asset_type, "rank_label": label,
                    "island_id": row["island_id"], "blocking_label": "Operational", "count": 0,
                })

        if rest_island_ids:
            rest_rows = sub_blocking[sub_blocking["island_id"].isin(rest_island_ids)]
            rest_agg = rest_rows.groupby("blocking_label", observed=True)["count"].sum().reset_index()
            for _, brow in rest_agg.iterrows():
                rows.append({
                    "timestep": timestep, "type": asset_type, "rank_label": "Other islands",
                    "island_id": np.nan, "blocking_label": brow["blocking_label"], "count": brow["count"],
                })

        unassigned_rows = sub_blocking[sub_blocking["island_id"].isna()]
        if not unassigned_rows.empty:
            unassigned_agg = unassigned_rows.groupby("blocking_label", observed=True)["count"].sum().reset_index()
            for _, brow in unassigned_agg.iterrows():
                rows.append({
                    "timestep": timestep, "type": asset_type, "rank_label": "Unassigned (no island match)",
                    "island_id": np.nan, "blocking_label": brow["blocking_label"], "count": brow["count"],
                })

    return pd.DataFrame(rows)


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def build_summary_text(trace_df: pd.DataFrame, timestep: int, asset_type: str, top_n: int = 2) -> str:
    """'<label>: N islands hold assets; the largest TOP_N hold P% of
    <type>, of which Q% are operational.' -- the narrative callout the
    panel is built to answer directly. The timestep itself is deliberately
    omitted: it is shown once, by the slider, and repeating it per
    asset-type subplot is redundant and wastes vertical space."""
    sub = trace_df[(trace_df["timestep"] == timestep) & (trace_df["type"] == asset_type)]
    label = ASSET_TYPE_LABELS.get(asset_type, asset_type)
    if sub.empty:
        return f"{label}: no assets recorded."
    n_islands = sub["island_id"].nunique()
    totals = sub.groupby("island_id", observed=True).size().sort_values(ascending=False)
    top_total = totals.iloc[:top_n].sum()
    # Use the true total asset count (including any with no island match --
    # see the "Unassigned (no island match)" bucket in build_ranked_island_
    # frame) rather than only summing island-matched totals, so pct_held is
    # not silently inflated by excluding unassigned assets from the base.
    grand_total = len(sub)
    pct_held = 100.0 * top_total / grand_total if grand_total else 0.0
    top_island_ids = totals.index[:top_n].tolist()
    top_sub = sub[sub["island_id"].isin(top_island_ids)]
    pct_operational = 100.0 * top_sub["operational"].astype(bool).mean() if len(top_sub) else 0.0
    method = _VORONOI_METHOD_BY_FUNCTION.get(asset_type)
    # Annotations do not auto-wrap within a subplot column, so the body text
    # is wrapped explicitly into short lines (narrow columns -> narrow wrap
    # width) rather than left to overflow into neighbouring columns.
    body = (
        f"{n_islands} islands; largest {top_n} hold {pct_held:.0f}% "
        f"({int(top_total)}/{int(grand_total)}), {pct_operational:.0f}% operational."
    )
    lines = [f"<b>{label}</b>"] + textwrap.wrap(body, width=20)
    if method:
        note = f"(via {method.replace('_', ' ')}, not island reachability)"
        lines += textwrap.wrap(note, width=24)
    return "<br>".join(lines)


# --------------------------------------------------------------------------
# Cheap aggregate across ALL discovered draws: "communicate uncertainty"
# --------------------------------------------------------------------------
# Qualitative palette reused across the uncertainty-lines panel; one color
# per policy (not per draw), so multiple scenarios of the same policy share
# a color at low opacity -- an envelope-by-eye, without the cost of
# actually computing/rendering a percentile band.
_POLICY_COLOR_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def build_pct_operational_timeseries(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-draw aggregate: % of assets operational, by (type,
    timestep). Asset-count-weighted (not population-weighted) -- a coarse
    'how much does this vary across draws' signal, not a substitute for the
    population-weighted access metrics in Figures 2-4."""
    agg = (
        trace_df.groupby(["type", "timestep"], observed=True)["operational"]
        .mean()
        .mul(100.0)
        .rename("pct_operational")
        .reset_index()
    )
    return agg


def build_uncertainty_lines_figure(
    runs_with_data: List[Tuple["TraceRun", pd.DataFrame]], asset_types: List[str],
) -> go.Figure:
    """One line per captured draw (any number -- this is the cheap panel
    meant to scale to many scenarios), grouped/colored by policy, showing
    % operational over time for each asset type. Hover reveals exactly
    which draw a line belongs to (policy, scenario, sampled uncertainty
    values), answering "what draw am I looking at?"."""
    n_cols = min(3, len(asset_types)) or 1
    n_rows = -(-len(asset_types) // n_cols)
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[ASSET_TYPE_LABELS.get(t, t) for t in asset_types],
    )

    policy_order = sorted({run.policy for run, _ in runs_with_data})
    policy_color = {p: _POLICY_COLOR_PALETTE[i % len(_POLICY_COLOR_PALETTE)] for i, p in enumerate(policy_order)}
    legend_shown = set()

    # Compute each run's aggregate once (not once per asset type).
    run_aggs = [(run, build_pct_operational_timeseries(trace_df)) for run, trace_df in runs_with_data]

    for i, asset_type in enumerate(asset_types):
        row, col = (i // n_cols) + 1, (i % n_cols) + 1
        for run, agg in run_aggs:
            sub = agg[agg["type"] == asset_type].sort_values("timestep")
            if sub.empty:
                continue
            color = policy_color[run.policy]
            show_legend = run.policy not in legend_shown
            legend_shown.add(run.policy)
            fig.add_trace(
                go.Scatter(
                    x=sub["timestep"], y=sub["pct_operational"], mode="lines",
                    line=dict(color=color, width=1.5), opacity=0.7,
                    name=run.policy, legendgroup=run.policy, showlegend=show_legend and i == 0,
                    hovertext=run.hover_text, hoverinfo="text+x+y",
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title="Uncertainty across draws: % operational by asset type, by timestep "
              "(one line per captured scenario, colored by policy)",
        height=280 * n_rows,
        legend=dict(title="Policy", orientation="h", y=-0.15),
    )
    fig.update_xaxes(title_text="Timestep (hour)")
    fig.update_yaxes(title_text="% operational", range=[0, 100])
    return fig


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def build_figure(ranked_df: pd.DataFrame, trace_df: pd.DataFrame, asset_types: List[str],
                  title_suffix: str = "") -> go.Figure:
    timesteps = sorted(ranked_df["timestep"].unique())
    rank_order = ["Largest island", "2nd largest", "3rd largest", "4th largest", "5th largest",
                  "6th largest", "7th largest", "8th largest", "9th largest", "10th largest",
                  "Other islands", "Unassigned (no island match)"]
    blocking_order = ["Operational", "Physically damaged", "Hazard-blocked", "Dependency-blocked", "Other blocked"]

    n_types = len(asset_types)
    fig = go.Figure()
    # One stacked-bar trace per (asset_type, blocking_label); traces are
    # indexed consistently across all frames so Plotly can animate them.
    trace_index: Dict[Tuple[str, str], int] = {}
    t0 = timesteps[0]
    idx = 0
    for ti, asset_type in enumerate(asset_types):
        sub0 = ranked_df[(ranked_df["timestep"] == t0) & (ranked_df["type"] == asset_type)]
        x_categories = [r for r in rank_order if r in sub0["rank_label"].unique()]
        for blocking_label in blocking_order:
            vals = [
                sub0[(sub0["rank_label"] == r) & (sub0["blocking_label"] == blocking_label)]["count"].sum()
                for r in x_categories
            ]
            fig.add_trace(go.Bar(
                x=x_categories, y=vals, name=blocking_label,
                marker_color=BLOCKING_REASON_COLORS[blocking_label],
                legendgroup=blocking_label, showlegend=(ti == 0),
                xaxis=f"x{ti + 1}" if ti else "x", yaxis=f"y{ti + 1}" if ti else "y",
            ))
            trace_index[(asset_type, blocking_label)] = idx
            idx += 1

    # Layout: one subplot column per asset type. The y-domain is deliberately
    # capped below 1.0 (not the full height) so the per-subplot summary
    # annotations have their own reserved band above the bars instead of
    # floating on top of them.
    axis_width = 1.0 / n_types
    annotation_y = 0.70
    plot_y_domain = [0.0, 0.68]
    layout_updates = {}
    subplot_x_centers: List[float] = []
    for ti, asset_type in enumerate(asset_types):
        label = ASSET_TYPE_LABELS.get(asset_type, asset_type)
        domain = [ti * axis_width + 0.02, (ti + 1) * axis_width - 0.02]
        subplot_x_centers.append(sum(domain) / 2.0)
        xaxis_key = "xaxis" if ti == 0 else f"xaxis{ti + 1}"
        yaxis_key = "yaxis" if ti == 0 else f"yaxis{ti + 1}"
        layout_updates[xaxis_key] = dict(domain=domain, title=label, tickangle=45)
        layout_updates[yaxis_key] = dict(
            title="Asset count" if ti == 0 else None, anchor=xaxis_key.replace("axis", ""),
            domain=plot_y_domain,
        )
    fig.update_layout(**layout_updates)

    def _summary_annotations(t: int) -> List[dict]:
        return [
            dict(
                text=build_summary_text(trace_df, t, at), showarrow=False,
                xref="paper", yref="paper", x=subplot_x_centers[ti], y=annotation_y,
                xanchor="center", yanchor="bottom", align="center", font=dict(size=9),
            )
            for ti, at in enumerate(asset_types)
        ]

    # Frames: one per timestep, updating every trace's y-values.
    frames = []
    for t in timesteps:
        frame_data = []
        for asset_type in asset_types:
            sub = ranked_df[(ranked_df["timestep"] == t) & (ranked_df["type"] == asset_type)]
            x_categories = [r for r in rank_order if r in sub["rank_label"].unique()]
            for blocking_label in blocking_order:
                vals = [
                    sub[(sub["rank_label"] == r) & (sub["blocking_label"] == blocking_label)]["count"].sum()
                    for r in x_categories
                ]
                frame_data.append(go.Bar(x=x_categories, y=vals))
        frames.append(go.Frame(
            data=frame_data, name=str(t),
            layout=go.Layout(annotations=_summary_annotations(t)),
        ))
    fig.frames = frames

    title = "Per-island asset composition and operational status, by timestep"
    if title_suffix:
        title += f" -- {title_suffix}"
    fig.update_layout(
        title=title,
        barmode="stack",
        height=640,
        margin=dict(t=90, b=210, l=60, r=30),
        legend=dict(title="Status", orientation="h", x=0.5, xanchor="center", y=-0.42, yanchor="top"),
        annotations=_summary_annotations(t0),
        sliders=[dict(
            active=0,
            x=0.12, len=0.82, xanchor="left",
            y=-0.68, yanchor="top",
            pad=dict(t=20),
            currentvalue=dict(prefix="Timestep (hour): "),
            steps=[
                dict(method="animate", label=str(t),
                     args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                for t in timesteps
            ],
        )],
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.0, xanchor="left", y=-0.68, yanchor="top",
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)]),
                     dict(label="Pause", method="animate",
                          args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])],
        )],
    )
    return fig


# --------------------------------------------------------------------------
# Optional context panel: existing Figure 2-4 summary-stats CSVs, if present
# --------------------------------------------------------------------------
def build_context_html() -> str:
    csv_names = [
        ("figure2_loss_of_access_stats.csv", "Figure 2 -- Loss of access by population group"),
        ("figure3_policy_benefit_stats.csv", "Figure 3 -- Adaptation benefits by population group"),
        ("figure4_policy_rankings.csv", "Figure 4 -- Policy rankings by population group"),
    ]
    blocks = []
    for filename, title in csv_names:
        path = FIG_DIR / filename
        if not path.exists():
            blocks.append(f"<h3>{title}</h3><p><em>Not found ({filename}) -- run "
                           f"produce_figures_group_equity.py first to generate it.</em></p>")
            continue
        df = pd.read_csv(path)
        blocks.append(f"<h3>{title}</h3>" + df.head(20).to_html(index=False, border=0))
    return "<hr>".join(blocks)


def main():
    args = parse_args()

    if args.trace_path is not None:
        m = _TRACE_FILENAME_RE.match(args.trace_path.name)
        if m:
            runs = [TraceRun(args.trace_path, m.group("execution_id"), m.group("policy"), m.group("scenario"))]
        else:
            runs = [TraceRun(args.trace_path, "?", args.trace_path.stem, "?")]
    else:
        policies = args.policies.split(",") if args.policies else None
        runs = discover_traces(args.trace_dir, policies, execution_id=args.execution_id)

    if not runs:
        wanted = args.policies or DEFAULT_POLICY
        exec_note = f" (execution_id={args.execution_id})" if args.execution_id else ""
        print(
            f"No spatial-trace parquet found (looked for policies: {wanted}{exec_note}) under {args.trace_dir}.\n"
            "Capture one first with e.g.:\n"
            "  python book/interdependent_ci_societal_access.py --scenarios 1 "
            f"--spatial-trace-policies {DEFAULT_POLICY} --spatial-trace-scenarios 1"
        )
        sys.exit(1)

    print(f"Found {len(runs)} run(s): " + ", ".join(r.label for r in runs))

    # Load every discovered trace once; reused both for the cheap
    # "uncertainty across draws" panel (all runs) and the expensive
    # per-island detail panel (curated subset only, see below).
    loaded = []  # List[Tuple[TraceRun, pd.DataFrame]]
    skipped = []
    for run in runs:
        print(f"Loading spatial trace: {run.path}")
        try:
            trace_df = load_trace(run.path)
        except Exception as exc:
            print(f"  SKIPPED ({exc})")
            skipped.append((run, str(exc)))
            continue
        asset_types = sorted(trace_df["type"].unique())
        print(f"  Asset types: {asset_types}; timesteps {trace_df['timestep'].min()}-{trace_df['timestep'].max()}")
        loaded.append((run, trace_df))
    if skipped:
        print(f"Skipped {len(skipped)} unusable trace(s) (e.g. captured before a required field "
              f"was added -- recapture if you need them): " + "; ".join(f"{r.label}: {msg}" for r, msg in skipped))
    if not loaded:
        print("All discovered traces were unusable; nothing to render.")
        sys.exit(1)

    all_asset_types = sorted({t for _, df in loaded for t in df["type"].unique()})
    print("Building uncertainty-across-draws panel (all loaded runs)...")
    lines_fig = build_uncertainty_lines_figure(loaded, all_asset_types)
    lines_html = lines_fig.to_html(include_plotlyjs="cdn", full_html=False)

    # Curated subset for the expensive full per-island breakdown.
    if args.detail_policies:
        wanted = {p.strip() for p in args.detail_policies.split(",") if p.strip()}
        detail_runs = [(run, df) for run, df in loaded if run.policy in wanted]
    else:
        detail_runs = loaded[: args.max_detail_runs]
    if len(loaded) > len(detail_runs):
        print(f"Full per-island detail limited to {len(detail_runs)}/{len(loaded)} run(s) "
              "(the expensive panel) -- use --detail-policies/--max-detail-runs to change this; "
              "all runs are still shown in the uncertainty-across-draws panel above.")

    run_blocks = []
    for run, trace_df in detail_runs:
        asset_types = sorted(trace_df["type"].unique())
        ranked_df = build_ranked_island_frame(trace_df, top_n=args.top_n_islands)
        fig = build_figure(ranked_df, trace_df, asset_types, title_suffix=run.label)
        i = len(run_blocks)
        fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
        display = "block" if i == 0 else "none"
        uncertainty_note = (
            "<br>".join(f"{k}: {v}" for k, v in sorted(run.uncertainties.items()))
            if run.uncertainties else "<em>no uncertainty metadata (older capture)</em>"
        )
        run_blocks.append(
            f'<div class="run-panel" id="run-panel-{i}" style="display:{display}">'
            f"<p>Source trace: {run.path.name}<br>{uncertainty_note}</p>"
            f"{fig_html}"
            "</div>"
        )
        run.panel_index = i

    selector_options = "\n".join(
        f'<option value="{run.panel_index}">{run.label}</option>' for run, _ in detail_runs
    )
    selector_html = (
        "<p><label for='run-selector'><strong>Deep-dive run: </strong></label>"
        f"<select id='run-selector' onchange=\"document.querySelectorAll('.run-panel')"
        ".forEach(function(d){d.style.display='none';});"
        "document.getElementById('run-panel-'+this.value).style.display='block';\">"
        f"{selector_options}</select></p>"
        if len(detail_runs) > 1 else ""
    )

    context_html = build_context_html()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    full_html = (
        "<html><head><title>Island asset explorer</title></head><body>"
        "<h1>Island asset explorer</h1>"
        "<h2>Uncertainty across draws</h2>"
        f"{lines_html}"
        "<h2>Per-island deep dive (single draw)</h2>"
        f"{selector_html}"
        + "".join(run_blocks) +
        "<h2>Ensemble context (Figures 2-4, if available)</h2>"
        f"{context_html}"
        "</body></html>"
    )
    args.output_path.write_text(full_html, encoding="utf-8")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
