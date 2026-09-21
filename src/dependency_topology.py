"""Runtime topology expansion for type-level dependency rules.

:mod:`src.dependency_knowledge_graph` defines *static, type-level* dependency
rules (``source_type`` -> ``target_type`` via a ``topology`` and an
``availability_policy``). This module expands those type-level rules into
concrete, asset-level :class:`DependencyEdge` records once ``gdf_assets`` is
known. Asset indices only ever appear in these runtime-generated structures --
never in the static knowledge-graph configuration.

Runtime edge representation
----------------------------
Each dependency rule expands into exactly one :class:`DependencyEdge` per
target asset (a "target-centric edge group"). This single representation is
used for every topology and captures three distinguishable situations for a
given rule (see :func:`expand_dependency_edges`):

* **No target assets of ``target_type`` exist**: the rule contributes no
  edges at all -- there is nothing to evaluate.
* **Target assets exist but no qualifying provider was found**: an edge is
  still produced, with ``provider_indices == ()``. This is an explicit
  "dependency unavailable" result, distinct from "nothing to evaluate".
* **Target assets exist with one or more qualifying providers**: an edge is
  produced with ``provider_indices`` populated (deterministically ordered),
  ready for policy evaluation (``exclusive`` / ``any`` / ``at_least_n``) in a
  later slice.

No provider availability (operational state) is evaluated in this module --
only spatial/topological qualification. That is left to the evaluator slice.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.dependency_knowledge_graph import (
    KnowledgeGraphRule,
    POLICY_AT_LEAST_N,
    TOPOLOGY_DIRECT,
    TOPOLOGY_RADIUS,
    TOPOLOGY_VORONOI,
)
from src.utils import create_spatial_index

# Floating-point tolerance (metres) applied to radius-boundary distance
# comparisons so that providers sitting exactly on the radius boundary are
# not excluded due to CRS-transform or geometry rounding error.
RADIUS_DISTANCE_TOLERANCE_M = 1e-6


@dataclass(frozen=True)
class DependencyEdge:
    """A concrete, asset-level dependency edge group for a single target asset.

    Attributes:
        edge_key: A stable, deterministic identifier for this edge group,
            derived from the rule identity and the target asset index. Safe
            to use as a dict key across repeated expansions of the same
            knowledge graph and asset frame.
        target_index: The positional index of the dependent/target asset in
            ``gdf_assets``.
        target_type: The asset type of the target.
        source_type: The asset type of the provider(s).
        relation: Always ``"dependency"`` (carried through from the static
            rule for traceability).
        topology: ``"direct"``, ``"voronoi"``, or ``"radius"``.
        availability_policy: ``"exclusive"``, ``"any"``, or ``"at_least_n"``.
        minimum_available: Set only when ``availability_policy ==
            "at_least_n"``; ``None`` otherwise.
        provider_indices: Deterministically ordered tuple of positional
            provider asset indices that qualify for this target under the
            rule's topology. An **empty tuple** means the target has no
            qualifying provider (an explicit "dependency unavailable"
            result) -- it does not mean the rule does not apply.
    """

    edge_key: str
    target_index: int
    target_type: str
    source_type: str
    relation: str
    topology: str
    availability_policy: str
    minimum_available: int | None
    provider_indices: tuple[int, ...]

    @property
    def has_provider(self) -> bool:
        """``True`` if at least one provider qualifies for this target."""
        return len(self.provider_indices) > 0


def _build_edge_key(rule: KnowledgeGraphRule, target_index: int) -> str:
    """Build a stable, collision-resistant edge key for one (rule, target) pair.

    Includes every field that distinguishes one dependency rule from another
    -- ``relation``, ``source_type``/``target_type``, ``topology``,
    ``availability_policy``, and (only when applicable) ``minimum_available``
    -- plus the target asset index. Two rules that differ only in
    ``relation`` or ``minimum_available`` therefore never collide. The key is
    independent of runtime provider availability (it never encodes
    ``provider_indices``), so it stays stable across timesteps even as
    provider availability changes.
    """
    parts = [
        rule.relation,
        f"{rule.source_type}->{rule.target_type}",
        rule.topology,
        rule.availability_policy,
    ]
    if rule.availability_policy == POLICY_AT_LEAST_N:
        parts.append(f"min={rule.minimum_available}")
    parts.append(f"target={target_index}")
    return "|".join(parts)


def _make_edge(
    rule: KnowledgeGraphRule,
    target_index: int,
    provider_indices: tuple[int, ...],
) -> DependencyEdge:
    return DependencyEdge(
        edge_key=_build_edge_key(rule, target_index),
        target_index=int(target_index),
        target_type=rule.target_type,
        source_type=rule.source_type,
        relation=rule.relation,
        topology=rule.topology,
        availability_policy=rule.availability_policy,
        minimum_available=rule.minimum_available,
        provider_indices=tuple(sorted(int(p) for p in provider_indices)),
    )


def _type_indices(gdf_assets, asset_type: str) -> list[int]:
    mask = gdf_assets["type"] == asset_type
    return sorted(int(i) for i in gdf_assets.index[mask])


# ---------------------------------------------------------------------------
# Direct topology
# ---------------------------------------------------------------------------
def _expand_direct_topology(gdf_assets, rule: KnowledgeGraphRule) -> list[DependencyEdge]:
    """Expand a ``topology="direct"`` rule.

    "Direct" topology has no inherent geometric assignment logic (that is
    what ``"voronoi"`` and ``"radius"`` are for). Its deterministic,
    documented behaviour is:

    * No target assets of ``target_type`` -> no edges at all.
    * No source assets of ``source_type`` -> one edge per target with an
      empty ``provider_indices`` (explicit "unavailable").
    * Exactly one source asset of ``source_type`` -> that single asset
      governs every target of ``target_type`` (unambiguous 1:many mapping,
      matching the existing single-primary-asset convention).
    * More than one source asset of ``source_type`` -> the mapping is
      ambiguous under "direct" topology, so a :class:`ValueError` is raised
      rather than silently guessing. Callers with multiple providers should
      use ``topology="voronoi"`` or ``topology="radius"`` for spatial
      assignment.
    """
    target_indices = _type_indices(gdf_assets, rule.target_type)
    if not target_indices:
        return []

    source_indices = _type_indices(gdf_assets, rule.source_type)
    if not source_indices:
        return [_make_edge(rule, t, ()) for t in target_indices]

    if len(source_indices) > 1:
        raise ValueError(
            "topology='direct' cannot deterministically map "
            f"{len(source_indices)} source assets of type '{rule.source_type}' "
            f"to target type '{rule.target_type}'. Use topology='voronoi' or "
            "topology='radius' for spatial assignment among multiple "
            "providers, or ensure exactly one source asset of this type "
            "exists."
        )

    provider = source_indices[0]
    return [_make_edge(rule, t, (provider,)) for t in target_indices]


# ---------------------------------------------------------------------------
# Voronoi topology
# ---------------------------------------------------------------------------
def _assign_targets_to_voronoi_providers(voronoi_gdf, target_gdf) -> dict[int, int | None]:
    """Assign each target to at most one Voronoi provider, with no fallback.

    Deliberately omits the nearest-neighbour fallback used by the legacy
    ``build_voronoi_service_area_map`` helper: an exclusive Voronoi mapping
    must never silently substitute a nearby alternate provider. Targets that
    cannot be resolved by overlap or point-in-polygon containment are
    assigned ``None``.
    """
    assignments: dict[int, int | None] = {}
    if voronoi_gdf.empty:
        return {int(idx): None for idx in target_gdf.index}

    sindex = create_spatial_index(voronoi_gdf)
    records = {
        int(idx): (int(asset_id), geom)
        for idx, asset_id, geom in voronoi_gdf.itertuples(index=True, name=None)
    }

    for target_idx, row in target_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            assignments[int(target_idx)] = None
            continue

        point = geom.centroid if hasattr(geom, "centroid") else geom
        best_label = None
        best_overlap = -1.0

        for label in sindex.intersection(geom.bounds):
            record = records.get(int(label))
            if record is None:
                continue
            _, vor_geom = record
            if vor_geom is None or vor_geom.is_empty or not vor_geom.intersects(geom):
                continue
            try:
                overlap = vor_geom.intersection(geom).area
            except Exception:
                overlap = 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = int(label)

        if best_label is None:
            for label in sindex.intersection(point.bounds):
                record = records.get(int(label))
                if record is None:
                    continue
                _, vor_geom = record
                if vor_geom is not None and not vor_geom.is_empty and (
                    vor_geom.contains(point) or vor_geom.intersects(point)
                ):
                    best_label = int(label)
                    break

        assignments[int(target_idx)] = (
            records[best_label][0] if best_label is not None else None
        )

    return assignments


def _expand_voronoi_topology(gdf_assets, rule: KnowledgeGraphRule) -> list[DependencyEdge]:
    """Expand a ``topology="voronoi"`` rule.

    Each target is assigned to exactly one governing provider (the Voronoi
    cell containing it), with no fallback -- if no cell can be resolved for a
    target, its edge has an empty ``provider_indices``. Voronoi topology
    therefore always yields 0 or 1 providers per target by construction,
    regardless of ``availability_policy``.
    """
    target_indices = _type_indices(gdf_assets, rule.target_type)
    if not target_indices:
        return []

    source_indices = _type_indices(gdf_assets, rule.source_type)
    if not source_indices:
        return [_make_edge(rule, t, ()) for t in target_indices]

    from src.impacts import create_voronoi_for_asset_type

    voronoi_gdf = create_voronoi_for_asset_type(gdf_assets, rule.source_type)
    target_gdf = gdf_assets.loc[target_indices]
    if (
        voronoi_gdf.crs is not None
        and target_gdf.crs is not None
        and target_gdf.crs != voronoi_gdf.crs
    ):
        target_gdf = target_gdf.to_crs(voronoi_gdf.crs)

    assignments = _assign_targets_to_voronoi_providers(voronoi_gdf, target_gdf)

    edges = []
    for target_idx in target_indices:
        provider = assignments.get(target_idx)
        providers: tuple[int, ...] = (provider,) if provider is not None else ()
        edges.append(_make_edge(rule, target_idx, providers))
    return edges


# ---------------------------------------------------------------------------
# Radius topology
# ---------------------------------------------------------------------------
def _crs_is_metric(crs) -> bool:
    try:
        axis_units = {
            axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name
        }
    except Exception:
        return False
    return bool(axis_units) and axis_units.issubset({"metre", "meter"})


def _ensure_projected_metric_gdf(gdf_assets):
    """Return *gdf_assets* in a projected metric CRS, or raise a clear ValueError.

    * ``gdf_assets.crs`` is ``None`` -> raise (radius distances are undefined
      without a CRS).
    * ``gdf_assets.crs`` is already projected and metric -> used as-is.
    * ``gdf_assets.crs`` is geographic -> reproject using
      :meth:`geopandas.GeoDataFrame.estimate_utm_crs`, a reliable UTM-zone
      estimator based on the data's own extent.
    * ``gdf_assets.crs`` is projected but not metric (e.g. US survey feet),
      or no suitable projected metric CRS could be estimated -> raise.
    """
    crs = gdf_assets.crs
    if crs is None:
        raise ValueError(
            "Radius topology requires gdf_assets.crs to be set, but it is "
            "None. A projected metric CRS (e.g. a UTM zone) is required to "
            "compute radius_m distances."
        )

    if crs.is_projected and _crs_is_metric(crs):
        return gdf_assets

    if crs.is_geographic:
        try:
            estimated_crs = gdf_assets.estimate_utm_crs()
        except Exception as exc:
            raise ValueError(
                "Radius topology could not derive a projected metric CRS "
                f"from gdf_assets.crs={crs!r}. Reproject gdf_assets to a "
                "projected metric CRS (e.g. a UTM zone) before expanding "
                "radius topology."
            ) from exc
        if estimated_crs is None or not _crs_is_metric(estimated_crs):
            raise ValueError(
                "Radius topology could not derive a projected metric CRS "
                f"from gdf_assets.crs={crs!r}. Reproject gdf_assets to a "
                "projected metric CRS (e.g. a UTM zone) before expanding "
                "radius topology."
            )
        return gdf_assets.to_crs(estimated_crs)

    raise ValueError(
        f"Radius topology requires a projected metric CRS, but gdf_assets.crs="
        f"{crs!r} is projected in non-metre units. Reproject gdf_assets to a "
        "projected metric CRS (e.g. a UTM zone) before expanding radius "
        "topology."
    )


def _expand_radius_topology(gdf_assets, rule: KnowledgeGraphRule) -> list[DependencyEdge]:
    """Expand a ``topology="radius"`` rule.

    Every source asset within ``radius_m`` of a target (inclusive of the
    boundary, subject to :data:`RADIUS_DISTANCE_TOLERANCE_M`) qualifies as a
    provider for that target. Distance is computed with an explicit
    ``distance() <= radius_m`` comparison in a projected metric CRS -- not a
    buffer/``within`` predicate, which would only approximate the circle and
    could exclude exact-boundary providers.
    """
    target_indices = _type_indices(gdf_assets, rule.target_type)
    if not target_indices:
        return []

    source_indices = _type_indices(gdf_assets, rule.source_type)
    if not source_indices:
        return [_make_edge(rule, t, ()) for t in target_indices]

    metric_gdf = _ensure_projected_metric_gdf(gdf_assets)
    source_geoms = {i: metric_gdf.geometry.loc[i] for i in source_indices}

    edges = []
    for target_idx in target_indices:
        target_geom = metric_gdf.geometry.loc[target_idx]
        providers: list[int] = []
        if target_geom is not None and not target_geom.is_empty:
            for source_idx in source_indices:
                source_geom = source_geoms[source_idx]
                if source_geom is None or source_geom.is_empty:
                    continue
                distance = source_geom.distance(target_geom)
                if distance <= rule.radius_m + RADIUS_DISTANCE_TOLERANCE_M:
                    providers.append(source_idx)
        edges.append(_make_edge(rule, target_idx, tuple(providers)))
    return edges


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
_TOPOLOGY_EXPANDERS = {
    TOPOLOGY_DIRECT: _expand_direct_topology,
    TOPOLOGY_VORONOI: _expand_voronoi_topology,
    TOPOLOGY_RADIUS: _expand_radius_topology,
}


def expand_dependency_edges(gdf_assets, knowledge_graph) -> list[DependencyEdge]:
    """Expand every dependency rule in *knowledge_graph* into runtime edges.

    Args:
        gdf_assets: GeoDataFrame of all assets, indexed positionally (as
            produced by :func:`src.utils.compile_asset_gdfs`), with a
            ``'type'`` column.
        knowledge_graph: A
            :class:`~src.dependency_knowledge_graph.DependencyKnowledgeGraph`.

    Returns:
        A flat, deterministically ordered list of :class:`DependencyEdge`
        objects -- one per (dependency rule, target asset) pair with at
        least one target asset of the matching type. Rules with zero
        matching targets contribute no edges.
    """
    if "type" not in gdf_assets.columns:
        raise ValueError("gdf_assets must contain a 'type' column.")

    edges: list[DependencyEdge] = []
    for rule in knowledge_graph.all_dependency_rules():
        expander = _TOPOLOGY_EXPANDERS.get(rule.topology)
        if expander is None:
            raise ValueError(f"Unsupported topology '{rule.topology}'.")
        edges.extend(expander(gdf_assets, rule))
    return edges


def index_edges_by_key(edges: list[DependencyEdge]) -> dict[str, DependencyEdge]:
    """Build a ``{edge_key: DependencyEdge}`` lookup for downstream use."""
    return {edge.edge_key: edge for edge in edges}


def index_edges_by_target(edges: list[DependencyEdge]) -> dict[int, list[DependencyEdge]]:
    """Group edges by ``target_index`` (a target may be governed by more than
    one dependency rule, e.g. two different source types)."""
    grouped: dict[int, list[DependencyEdge]] = {}
    for edge in edges:
        grouped.setdefault(edge.target_index, []).append(edge)
    return grouped


def targets_without_providers(edges: list[DependencyEdge]) -> list[int]:
    """Return the sorted, deterministic list of target indices with no
    qualifying provider (an explicit "dependency unavailable" result)."""
    return sorted({edge.target_index for edge in edges if not edge.has_provider})
