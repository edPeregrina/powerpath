# PowerPath module map

This document replaces the old root-level `MODULE_INFOGRAPHIC_MAP.md`.
It is a contributor-oriented map of the **current** architecture on the simplified knowledge-graph line of work.

## System overview

PowerPath simulates how flooding affects infrastructure assets over time, how damage and repair evolve, how road-network disruption changes access, and how those realized states are converted into societal-access and impact outputs.

Primary entry points:

- [`config.get_config()`](../config.py) builds the runtime configuration.
- [`load_electricity_assets()`](../src/data_loader.py) and [`load_hazard_maps()`](../src/data_loader.py) load the main inputs.
- [`simulate_asset_damage_recovery_access_breakdown()`](../src/simulation.py) runs the time-stepped simulation.
- [`postprocess_societal_access_results()`](../src/societal_access.py) adds societal-access metrics after the timestep loop.
- [`plot_ema_outcomes()`](../src/visualisations.py) and helpers in [`src/impacts.py`](../src/impacts.py) consume simulation outputs.

## Primary execution flow

```text
config.py
  -> data loading (assets, hazard maps)
  -> simulation initialization
      -> build knowledge graph
      -> expand dependency topology into runtime dependency edges
      -> load/build caches and adaptation inputs
  -> per timestep
      -> sample hazard at assets
      -> update road/island accessibility state
      -> update damage, repair timers, and intrinsic state
      -> evaluate hazard availability + dependency availability + restart waits
      -> assign crews and advance repairs
      -> record summary and detailed realized state
  -> postprocess societal access
      -> allocation cache lookup/build
      -> service-area population mapping for electricity-style services
      -> shared realized-state cache lookup/write
  -> impacts / visualization / notebook analysis
```

## Module responsibility map

| Concern | Main files | Current responsibility |
|---|---|---|
| Configuration and orchestration | [`config.py`](../config.py), [`src/simulation.py`](../src/simulation.py) | Defines runtime parameters, default knowledge-graph rules, cache/output locations, and the main simulation loop. |
| Input loading | [`src/data_loader.py`](../src/data_loader.py), [`src/hazard_analysis_electricity.py`](../src/hazard_analysis_electricity.py) | Loads asset shapefiles and hazard rasters, then extracts per-asset hazard values for each map. |
| Damage and recovery state | [`src/damage_recovery.py`](../src/damage_recovery.py), [`src/recovery_scheduler.py`](../src/recovery_scheduler.py) | Converts hazard intensity to damage and repair-time state, manages wait vectors, and supports fragility-based operational failure. |
| Knowledge-graph dependency model | [`src/dependency_knowledge_graph.py`](../src/dependency_knowledge_graph.py), [`src/dependency_topology.py`](../src/dependency_topology.py), [`src/dependency_evaluator.py`](../src/dependency_evaluator.py) | Separates type-level dependency rules from runtime edge expansion and from timestep-by-timestep operational evaluation. |
| Road topology and island access | [`src/island_analysis.py`](../src/island_analysis.py), [`src/utils.py`](../src/utils.py) | Filters the disrupted road graph, computes connected-component islands, maps assets to islands/access RFIDs, and redistributes island-based crews. |
| Adaptation and caches | [`src/adaptation.py`](../src/adaptation.py), [`src/caching.py`](../src/caching.py) | Builds adaptation depth reductions and manages persisted caches used during simulation/postprocessing. |
| Societal-access postprocessing | [`src/societal_access.py`](../src/societal_access.py), [`src/realized_state_cache.py`](../src/realized_state_cache.py) | Turns realized asset/island state into access metrics, handles allocation reuse, service-area overrides, and optional cross-experiment shared caching. |
| Impacts and visualization | [`src/impacts.py`](../src/impacts.py), [`src/visualisations.py`](../src/visualisations.py) | Derives impact summaries and plotting outputs from simulation and societal-access results. |
| Executable examples | [`book/`](../book/) | Notebooks demonstrate the current workflows, including knowledge-graph, societal-access, and realized-state-cache examples. |
| Regression coverage | [`tests/`](../tests/) | Focused tests validate dependency schema/topology/behavior, recovery semantics, island crew logic, societal-access behavior, and realized-state cache behavior. |

## Knowledge graph and dependency edges

The dependency architecture is intentionally split into three layers:

1. **Type-level rules** in [`src/dependency_knowledge_graph.py`](../src/dependency_knowledge_graph.py)
   - `KnowledgeGraphRule` stores either hazard rules or dependency rules.
   - `build_default_knowledge_graph()` defines the shipped flooding defaults.
   - `config.py` stores those defaults in `config['dependency_parameters']['knowledge_graph']`.
2. **Topology expansion** in [`expand_dependency_edges()`](../src/dependency_topology.py)
   - Expands type-to-type rules into concrete runtime `DependencyEdge` records once `gdf_assets` is known.
   - This is where `topology="direct"`, `"voronoi"`, and `"radius"` are resolved.
   - Expansion decides *which providers qualify for which targets*, not whether they are currently operating.
3. **Operational-state evaluation** in [`evaluate_operational_state()`](../src/dependency_evaluator.py)
   - Re-evaluates each timestep using current intrinsic state, hazard state, and dependency availability.
   - Produces the layered state used by the rest of the model:
     - `intrinsic_operational`
     - `hazard_available`
     - `dependency_available`
     - `restart_ready`
     - final `effective_operational`

That separation is important when changing the system:

- change [`src/dependency_knowledge_graph.py`](../src/dependency_knowledge_graph.py) when the **schema or defaults** change;
- change [`src/dependency_topology.py`](../src/dependency_topology.py) when the **provider-target matching logic** changes;
- change [`src/dependency_evaluator.py`](../src/dependency_evaluator.py) when the **runtime availability logic** changes.

### Current default concepts

From [`build_default_knowledge_graph()`](../src/dependency_knowledge_graph.py):

- flooding hazard rules exist for `msls`, `ms`, `ls`, and `hospital`;
- `msls -> hospital` is the active default dependency rule;
- the default dependency topology is `voronoi` with `availability_policy="exclusive"`, so each hospital is assigned exactly one governing substation at expansion time;
- roads are **not** part of the knowledge graph; road disruption is still handled by road-graph filtering and island analysis.

## End-to-end state flow

### 1. Configuration and loading

- [`config.get_config()`](../config.py) defines simulation, recovery, analysis, service-node, and dependency parameters.
- [`load_electricity_assets()`](../src/data_loader.py) assembles the infrastructure GeoDataFrame and assigns asset `type` values.
- [`load_hazard_maps()`](../src/data_loader.py) orders flood rasters for timestep processing.

### 2. Simulation initialization

[`_initialize_simulation()`](../src/simulation.py) prepares the runtime state:

- resolves directories and caches;
- expands `config['dependency_parameters']['knowledge_graph']` into runtime dependency edges if they were not precomputed;
- builds adaptation depth-reduction arrays when L1/L2 inputs are present;
- initializes the arrays later stored in `SimulationState`.

### 3. Hazard, damage, and repair progression

At each major timestep, [`_update_hazard_map_states()`](../src/simulation.py) calls [`find_hazard_value_at_points_optimized()`](../src/hazard_analysis_electricity.py) to sample the current hazard map at asset locations.

Damage/recovery semantics are split across the simulation and recovery modules:

- [`default_damage_ratio_function()`](../src/damage_recovery.py) maps hazard intensity to damage ratio.
- [`default_repair_time_function()`](../src/damage_recovery.py) maps damage ratio to repair time.
- [`default_fragility_function()`](../src/damage_recovery.py) samples operational failure from hazard intensity.
- repair countdowns live in `state.recovery_wait_vectors["repair_time"]`.

### 4. Repair and operational semantics

Current semantics to keep in mind:

- `state.intrinsic_operational` is the physical asset state.
- completed repairs clear damage and set `intrinsic_operational` back to `True` in [`_handle_completed_repairs()`](../src/simulation.py).
- final published `state.operational` is **not** the same thing as intrinsic state; it is the effective state after hazard and dependency evaluation.
- repairs only progress when the asset is:
  - assigned a crew,
  - accessible, and
  - **not currently flooded**.

That flooded-asset repair constraint is enforced in [`_update_repair_progress()`](../src/simulation.py), where repair timers decrement only for `accessible & ~flooded_mask & repair_crews_assigned` assets.

Hazard return-to-operational behavior comes from the knowledge graph:

- `repair_complete`: asset stays hazard-blocked until repair time reaches zero;
- `repair_below`: asset may return earlier once repair time drops below a threshold;
- `delayed`: asset waits on a named hazard-recovery countdown;
- dependency restart delays are tracked separately from hazard waits in [`src/dependency_evaluator.py`](../src/dependency_evaluator.py).

### 5. Dependency and topology evaluation

During the timestep loop, [`_update_operational_state()`](../src/simulation.py) calls [`evaluate_operational_state()`](../src/dependency_evaluator.py).

This step:

- combines intrinsic state with hazard rules;
- evaluates expanded dependency edges against current provider availability;
- iterates dependency chains to a fixed point;
- applies dependency restart delays;
- publishes the effective operational state used by downstream metrics and postprocessing.

### 6. Road/island access

If an island-based crew strategy is active:

- [`match_assets_access()`](../src/island_analysis.py) links assets to road-network access points;
- [`compute_island_geodataframe_from_graph()`](../src/island_analysis.py) derives disrupted road islands from the hazard-filtered graph;
- [`update_repair_crew_islands()`](../src/island_analysis.py) redistributes crews as island structure changes.

This road/island path is separate from the knowledge graph: it governs **reachability and crew logistics**, not service dependencies between asset types.

### 7. Societal-access postprocessing and shared cache path

Societal access is a postprocessing step, not part of the per-timestep damage solver.

[`postprocess_societal_access_results()`](../src/societal_access.py):

- reads timestep `operational`, `island_id`, and `road_state_key` outputs from the simulation;
- resolves or builds population-to-island allocations via [`get_or_build_allocation()`](../src/societal_access.py);
- builds island/function access metrics;
- overrides selected functions with service-area-based metrics when configured;
- optionally reuses flat scalar results from a shared realized-state cache.

For electricity-style service areas, [`_build_service_area_population_maps()`](../src/societal_access.py):

- precomputes provider-to-population assignments;
- uses Voronoi or nearest-provider logic depending on provider count;
- supports service-area-based overrides for functions such as electricity.

Shared realized-state caching lives in two places:

- [`build_shared_realized_state_cache_from_config()`](../src/realized_state_cache.py) creates the backend (currently SQLite-backed);
- [`postprocess_societal_access_results()`](../src/societal_access.py) builds realization-aware cache keys and reuses cached societal scalar fields when the realized state matches.

## Outputs and visualization

The main simulation returns:

- per-timestep summary metrics;
- per-timestep detailed asset state;
- final state arrays;
- updated caches.

Those outputs feed:

- [`src/impacts.py`](../src/impacts.py) for population/land-use/monetized impact calculations;
- [`src/visualisations.py`](../src/visualisations.py) for aggregated plots;
- notebooks in [`book/`](../book/) for reproducible workflows and interpretation.

## Where notebooks and tests fit

### Notebooks

The most relevant current notebooks are:

- [`book/Use_Case_sample_knowledge_graph.ipynb`](../book/Use_Case_sample_knowledge_graph.ipynb) — knowledge-graph dependency workflow;
- [`book/Use_Case_sample_societal_access.ipynb`](../book/Use_Case_sample_societal_access.ipynb) — societal-access workflow;
- [`book/realized_state_cache_demo.ipynb`](../book/realized_state_cache_demo.ipynb) — cross-experiment realized-state cache reuse.

Treat these as executable orientation material layered on top of the Python modules above, not as the canonical definition of architecture.

### Tests

Useful architectural test anchors include:

- [`tests/test_dependency_knowledge_graph_schema.py`](../tests/test_dependency_knowledge_graph_schema.py)
- [`tests/test_dependency_topology.py`](../tests/test_dependency_topology.py)
- [`tests/test_dependency_behaviors.py`](../tests/test_dependency_behaviors.py)
- [`tests/test_recovery_behaviors.py`](../tests/test_recovery_behaviors.py)
- [`tests/test_societal_access.py`](../tests/test_societal_access.py)
- [`tests/test_realized_state_cache.py`](../tests/test_realized_state_cache.py)
- [`tests/test_island_analysis_crew_redistribution.py`](../tests/test_island_analysis_crew_redistribution.py)

Use them to confirm expected contracts before changing the corresponding modules.

## How to navigate or change this system

When making changes, start from the layer you intend to affect:

- **New scenario or parameter wiring**: start in [`config.py`](../config.py) and [`src/simulation.py`](../src/simulation.py).
- **Hazard-to-damage or repair semantics**: start in [`src/damage_recovery.py`](../src/damage_recovery.py) and the repair/update helpers in [`src/simulation.py`](../src/simulation.py).
- **Dependency semantics**: inspect the trio of [`src/dependency_knowledge_graph.py`](../src/dependency_knowledge_graph.py), [`src/dependency_topology.py`](../src/dependency_topology.py), and [`src/dependency_evaluator.py`](../src/dependency_evaluator.py) together.
- **Road accessibility / island behavior**: inspect [`src/island_analysis.py`](../src/island_analysis.py).
- **Societal access or cache reuse**: inspect [`src/societal_access.py`](../src/societal_access.py) and [`src/realized_state_cache.py`](../src/realized_state_cache.py).
- **Example workflows**: inspect the notebooks in [`book/`](../book/).

A good contributor workflow is:

1. find the entry point used by the notebook or test you care about;
2. identify whether the behavior is configuration, topology expansion, runtime evaluation, or postprocessing;
3. update the matching focused tests first or alongside the code change;
4. only then widen to notebooks or higher-level workflows.

## Documentation maintenance

Update this map when any of the following change:

- the main simulation entry points or execution order;
- the knowledge-graph schema or default rules;
- dependency topology or operational-state layering;
- societal-access postprocessing, service-area overrides, or shared realized-state caching;
- the recommended contributor notebooks or test anchors;
- the location of this document or links to it.

Do **not** use this file as a backlog, migration scratchpad, or one-row-per-helper inventory. It should stay focused on the current architecture a new contributor needs to navigate safely.
