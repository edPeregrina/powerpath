"""Type-level dependency knowledge graph.

Defines two kinds of *type-level* rules, addressed purely by asset type
(never by asset ID or index):

* **Hazard rules** (``relation="hazard"``): describe how a given hazard type
  affects a single asset type -- whether the hazard directly blocks
  operation, and what must happen for the asset to return to operational.
* **Dependency rules** (``relation="dependency"``): describe how a
  ``source_type`` provides service to a ``target_type``, via a spatial
  ``topology`` (``"direct"``, ``"voronoi"``, or ``"radius"``) and an
  ``availability_policy`` (``"exclusive"``, ``"any"``, or ``"at_least_n"``)
  that determines how many available providers a target requires.

Runtime asset-level dependency *edges* (concrete provider/target asset index
pairs) are expanded from these type-level rules elsewhere; this module only
defines and validates the static, type-level configuration.

Usage example::

    from src.dependency_knowledge_graph import DependencyKnowledgeGraph

    kg = DependencyKnowledgeGraph.from_config([
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "msls",
            "hazard_blocks_operation": True,
            "return_to_operational": {"trigger": "repair_complete"},
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        },
    ])

    hazard_rules = kg.get_hazard_rules("flooding", "msls")
    dependency_rules = kg.get_dependency_rules(source_type="msls")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Return-to-operational trigger constants
# ---------------------------------------------------------------------------
TRIGGER_IMMEDIATE = "immediate"
"""Asset returns to operational as soon as hazard clears (no repair required)."""

TRIGGER_REPAIR_COMPLETE = "repair_complete"
"""Asset returns to operational only after repair_time reaches 0."""

TRIGGER_REPAIR_BELOW = "repair_below"
"""Asset returns to operational when repair_time < threshold (e.g. 2.0 hours)."""

TRIGGER_DELAYED = "delayed"
"""Asset returns to operational after a named wait vector has counted down to 0."""

VALID_TRIGGERS = {
    TRIGGER_IMMEDIATE,
    TRIGGER_REPAIR_COMPLETE,
    TRIGGER_REPAIR_BELOW,
    TRIGGER_DELAYED,
}

# ---------------------------------------------------------------------------
# Relation kinds
# ---------------------------------------------------------------------------
RELATION_HAZARD = "hazard"
"""A type-level rule describing how a hazard affects a single asset type."""

RELATION_DEPENDENCY = "dependency"
"""A type-level rule describing how a source type provides service to a target type."""

VALID_RELATIONS = {RELATION_HAZARD, RELATION_DEPENDENCY}

# ---------------------------------------------------------------------------
# Dependency topologies
# ---------------------------------------------------------------------------
TOPOLOGY_DIRECT = "direct"
"""Providers/targets are matched via precomputed or caller-supplied service areas."""

TOPOLOGY_VORONOI = "voronoi"
"""Targets are matched to the provider whose Voronoi cell contains them."""

TOPOLOGY_RADIUS = "radius"
"""Targets are matched to providers within ``radius_m`` (inclusive of the boundary)."""

VALID_TOPOLOGIES = {TOPOLOGY_DIRECT, TOPOLOGY_VORONOI, TOPOLOGY_RADIUS}

# ---------------------------------------------------------------------------
# Dependency availability policies
# ---------------------------------------------------------------------------
POLICY_EXCLUSIVE = "exclusive"
"""Topology assigns exactly one governing provider to a target; no fallback."""

POLICY_ANY = "any"
"""At least one qualifying provider among those matched by topology is required."""

POLICY_AT_LEAST_N = "at_least_n"
"""At least ``minimum_available`` qualifying providers are required."""

VALID_AVAILABILITY_POLICIES = {POLICY_EXCLUSIVE, POLICY_ANY, POLICY_AT_LEAST_N}


@dataclass
class ReturnToOperational:
    """Describes what must happen before an asset can return to operational state.

    Attributes:
        trigger: One of ``"immediate"``, ``"repair_complete"``, ``"repair_below"``,
            or ``"delayed"``.
        threshold: Only used when *trigger* is ``"repair_below"``; the asset
            becomes operational once ``repair_time < threshold``.
        delay_steps: Only used when *trigger* is ``"delayed"``.
        wait_vector: Wait-vector name used when *trigger* is ``"delayed"``.
    """

    trigger: str = TRIGGER_IMMEDIATE
    threshold: float = 0.0
    delay_steps: float = 0.0
    wait_vector: str = "dependency_wait"

    def __post_init__(self):
        if self.trigger not in VALID_TRIGGERS:
            raise ValueError(
                f"Invalid trigger '{self.trigger}'. Must be one of {VALID_TRIGGERS}."
            )
        if self.trigger == TRIGGER_REPAIR_BELOW and self.threshold <= 0.0:
            raise ValueError(
                "threshold must be > 0 when trigger is 'repair_below'."
            )
        if self.trigger == TRIGGER_DELAYED and self.delay_steps <= 0.0:
            raise ValueError(
                "delay_steps must be > 0 when trigger is 'delayed'."
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReturnToOperational:
        return cls(
            trigger=d.get("trigger", TRIGGER_IMMEDIATE),
            threshold=float(d.get("threshold", 0.0)),
            delay_steps=float(d.get("delay_steps", 0.0)),
            wait_vector=str(d.get("wait_vector", "dependency_wait")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"trigger": self.trigger}
        if self.trigger == TRIGGER_REPAIR_BELOW:
            result["threshold"] = self.threshold
        elif self.trigger == TRIGGER_DELAYED:
            result["delay_steps"] = self.delay_steps
            result["wait_vector"] = self.wait_vector
        return result


@dataclass
class KnowledgeGraphRule:
    """A single type-level rule: either a hazard rule or a dependency rule.

    Static configuration is addressed purely by asset *type* -- never by
    asset ID or index -- so a single rule automatically governs every asset
    of ``source_type`` (and, for dependency rules, every asset of
    ``target_type``) regardless of how many such assets exist.

    Attributes:
        relation: ``"hazard"`` or ``"dependency"``.
        source_type: The asset type this rule is anchored on. Required for
            both relations.
        hazard_type: The hazard type this rule applies to (e.g.
            ``"flooding"``). Required when ``relation == "hazard"``; not
            applicable to dependency rules.
        hazard_blocks_operation: If ``True``, assets of ``source_type`` are
            non-operational while hazard exposure exceeds the simulation
            flood threshold. Only applicable to hazard rules.
        return_to_operational: Specifies what triggers the return to
            operational. Only applicable to hazard rules.
        target_type: The asset type that depends on ``source_type``.
            Required when ``relation == "dependency"``; must be omitted for
            hazard rules.
        topology: ``"direct"``, ``"voronoi"``, or ``"radius"``. Required for
            dependency rules.
        availability_policy: ``"exclusive"``, ``"any"``, or ``"at_least_n"``.
            Required for dependency rules.
        radius_m: Search radius in metres (topology boundary is inclusive).
            Required only when ``topology == "radius"``.
        minimum_available: Minimum number of available providers required.
            Required only when ``availability_policy == "at_least_n"``.
    """

    relation: str
    source_type: str
    hazard_type: str | None = None
    hazard_blocks_operation: bool = False
    return_to_operational: ReturnToOperational | None = None
    target_type: str | None = None
    topology: str | None = None
    availability_policy: str | None = None
    radius_m: float | None = None
    minimum_available: int | None = None

    def __post_init__(self):
        if self.relation not in VALID_RELATIONS:
            raise ValueError(
                f"Invalid relation '{self.relation}'. Must be one of {VALID_RELATIONS}."
            )
        if not self.source_type:
            raise ValueError("source_type is required for every rule.")

        if self.relation == RELATION_HAZARD:
            self._validate_hazard_fields()
        else:
            self._validate_dependency_fields()

    def _validate_hazard_fields(self) -> None:
        if not self.hazard_type:
            raise ValueError(
                "hazard_type is required when relation is 'hazard'."
            )
        if self.target_type is not None:
            raise ValueError(
                "target_type must not be set when relation is 'hazard'. "
                "Hazard rules apply directly to source_type."
            )
        if self.topology is not None:
            raise ValueError(
                "topology is not applicable when relation is 'hazard'."
            )
        if self.availability_policy is not None:
            raise ValueError(
                "availability_policy is not applicable when relation is 'hazard'."
            )
        if self.radius_m is not None:
            raise ValueError(
                "radius_m is not applicable when relation is 'hazard'."
            )
        if self.minimum_available is not None:
            raise ValueError(
                "minimum_available is not applicable when relation is 'hazard'."
            )
        if self.return_to_operational is None:
            self.return_to_operational = ReturnToOperational()

    def _validate_dependency_fields(self) -> None:
        if not self.target_type:
            raise ValueError(
                "target_type is required when relation is 'dependency'."
            )
        if self.hazard_type is not None:
            raise ValueError(
                "hazard_type is not applicable when relation is 'dependency'. "
                "Dependency rules describe type-to-type service, not hazard exposure."
            )
        if self.return_to_operational is not None:
            raise ValueError(
                "return_to_operational is not applicable when relation is "
                "'dependency'."
            )
        if self.hazard_blocks_operation:
            raise ValueError(
                "hazard_blocks_operation is not applicable when relation is "
                "'dependency'."
            )

        if self.topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"Invalid topology '{self.topology}'. Must be one of {VALID_TOPOLOGIES}."
            )
        if self.availability_policy not in VALID_AVAILABILITY_POLICIES:
            raise ValueError(
                f"Invalid availability_policy '{self.availability_policy}'. "
                f"Must be one of {VALID_AVAILABILITY_POLICIES}."
            )

        if self.topology == TOPOLOGY_RADIUS:
            if self.radius_m is None or float(self.radius_m) <= 0.0:
                raise ValueError(
                    "radius_m must be a positive number when topology is 'radius'."
                )
            self.radius_m = float(self.radius_m)
        elif self.radius_m is not None:
            raise ValueError(
                "radius_m is only applicable when topology is 'radius'."
            )

        if self.availability_policy == POLICY_EXCLUSIVE and self.topology == TOPOLOGY_RADIUS:
            raise ValueError(
                "availability_policy='exclusive' requires a topology that "
                "assigns exactly one governing provider (topology='direct' "
                "or topology='voronoi'). topology='radius' does not "
                "guarantee a single provider -- any number of source assets "
                "within radius_m may qualify -- so it cannot be combined "
                "with an exclusive policy."
            )

        if self.availability_policy == POLICY_AT_LEAST_N:
            if self.minimum_available is None or int(self.minimum_available) < 1:
                raise ValueError(
                    "minimum_available must be a positive integer when "
                    "availability_policy is 'at_least_n'."
                )
            self.minimum_available = int(self.minimum_available)
        elif self.minimum_available is not None:
            raise ValueError(
                "minimum_available is only applicable when availability_policy "
                "is 'at_least_n'."
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeGraphRule:
        rto_dict = d.get("return_to_operational")
        return cls(
            relation=d.get("relation", RELATION_HAZARD),
            source_type=d.get("source_type"),
            hazard_type=d.get("hazard_type"),
            hazard_blocks_operation=bool(d.get("hazard_blocks_operation", False)),
            return_to_operational=(
                ReturnToOperational.from_dict(rto_dict) if rto_dict is not None else None
            ),
            target_type=d.get("target_type"),
            topology=d.get("topology"),
            availability_policy=d.get("availability_policy"),
            radius_m=d.get("radius_m"),
            minimum_available=d.get("minimum_available"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "relation": self.relation,
            "source_type": self.source_type,
        }
        if self.relation == RELATION_HAZARD:
            result["hazard_type"] = self.hazard_type
            result["hazard_blocks_operation"] = self.hazard_blocks_operation
            result["return_to_operational"] = self.return_to_operational.to_dict()
        else:
            result["target_type"] = self.target_type
            result["topology"] = self.topology
            result["availability_policy"] = self.availability_policy
            if self.topology == TOPOLOGY_RADIUS:
                result["radius_m"] = self.radius_m
            if self.availability_policy == POLICY_AT_LEAST_N:
                result["minimum_available"] = self.minimum_available
        return result

    @property
    def is_hazard_rule(self) -> bool:
        return self.relation == RELATION_HAZARD

    @property
    def is_dependency_rule(self) -> bool:
        return self.relation == RELATION_DEPENDENCY


class DependencyKnowledgeGraph:
    """Container for all type-level hazard and dependency rules in a simulation.

    Rules are addressed purely by asset type. Hazard rules are keyed by
    ``(hazard_type, source_type)``; dependency rules are keyed by
    ``(source_type, target_type)``. A single rule automatically governs
    every asset of the matching type(s), regardless of how many such assets
    exist -- there is no per-asset declaration in static configuration.

    Args:
        rules: Initial list of :class:`KnowledgeGraphRule` objects.
    """

    def __init__(self, rules: list[KnowledgeGraphRule] | None = None):
        self._rules: list[KnowledgeGraphRule] = rules or []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, rule_list: list[dict[str, Any]]) -> DependencyKnowledgeGraph:
        """Build a graph from a list of rule dicts (e.g. from ``config.py``)."""
        rules = [KnowledgeGraphRule.from_dict(r) for r in rule_list]
        return cls(rules)

    @classmethod
    def from_json(cls, json_str: str) -> DependencyKnowledgeGraph:
        """Deserialise from a JSON string."""
        return cls.from_config(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        """Serialise all rules to a JSON string."""
        return json.dumps(self.to_config(), indent=indent)

    def to_config(self) -> list[dict[str, Any]]:
        """Return rules in the configuration format accepted by the simulation."""
        return [rule.to_dict() for rule in self._rules]

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_hazard_rules(
        self,
        hazard_type: str,
        source_type: str,
    ) -> list[KnowledgeGraphRule]:
        """Return hazard rules matching the given hazard type and asset type."""
        return [
            r
            for r in self._rules
            if r.is_hazard_rule
            and r.hazard_type == hazard_type
            and r.source_type == source_type
        ]

    def get_default_hazard_rule(
        self, hazard_type: str, source_type: str
    ) -> KnowledgeGraphRule:
        """Return a no-op default hazard rule for types not present in the graph."""
        return KnowledgeGraphRule(
            relation=RELATION_HAZARD,
            source_type=source_type,
            hazard_type=hazard_type,
            hazard_blocks_operation=False,
            return_to_operational=ReturnToOperational(trigger=TRIGGER_IMMEDIATE),
        )

    def get_hazard_rules_or_default(
        self,
        hazard_type: str,
        source_type: str,
    ) -> list[KnowledgeGraphRule]:
        """Like :meth:`get_hazard_rules` but returns the default rule when none match."""
        matched = self.get_hazard_rules(hazard_type, source_type)
        if matched:
            return matched
        return [self.get_default_hazard_rule(hazard_type, source_type)]

    def get_dependency_rules(
        self,
        source_type: str | None = None,
        target_type: str | None = None,
    ) -> list[KnowledgeGraphRule]:
        """Return dependency rules, optionally filtered by source and/or target type.

        Passing ``None`` for either filter matches any value for that field.
        """
        return [
            r
            for r in self._rules
            if r.is_dependency_rule
            and (source_type is None or r.source_type == source_type)
            and (target_type is None or r.target_type == target_type)
        ]

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def all_hazard_rules(self) -> tuple[KnowledgeGraphRule, ...]:
        """Return all hazard rules in the graph."""
        return tuple(r for r in self._rules if r.is_hazard_rule)

    def all_dependency_rules(self) -> tuple[KnowledgeGraphRule, ...]:
        """Return all dependency rules in the graph."""
        return tuple(r for r in self._rules if r.is_dependency_rule)

    def all_source_types(self) -> list[str]:
        """Return the sorted unique set of all source_type values."""
        return sorted({r.source_type for r in self._rules})

    def all_target_types(self) -> list[str]:
        """Return the sorted unique set of all target_type values (dependency rules only)."""
        return sorted({r.target_type for r in self._rules if r.target_type is not None})

    def all_hazard_types(self) -> list[str]:
        """Return the sorted unique set of all hazard_type values."""
        return sorted({r.hazard_type for r in self._rules if r.hazard_type is not None})

    @property
    def rules(self) -> tuple[KnowledgeGraphRule, ...]:
        """Expose the configured rules as an immutable sequence."""
        return tuple(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"DependencyKnowledgeGraph({len(self._rules)} rules)"


# ---------------------------------------------------------------------------
# Default knowledge graph for existing electricity infrastructure assets
# ---------------------------------------------------------------------------

def build_default_knowledge_graph() -> DependencyKnowledgeGraph:
    """Return the baseline knowledge graph for flooding and the known asset types.

    Structural damage remains governed by fragility, while type-level hazard
    rules define when damaged assets may return to operation. The msls ->
    hospital dependency rule uses ``topology="voronoi"`` with
    ``availability_policy="exclusive"`` -- each hospital is assigned exactly
    one governing substation (its nearest msls Voronoi cell) at runtime
    expansion time, with no fallback provider. ``topology="direct"`` remains
    supported for callers with a single, unambiguous provider, but is not
    used as the default here because real datasets typically contain
    multiple msls substations and direct topology requires at most one
    source asset per rule.
    Roads are intentionally excluded because the existing road-graph exposure
    filtering governs their availability independently.

    Returns:
        A :class:`DependencyKnowledgeGraph` populated with sensible defaults.
    """
    rules = [
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "msls",
            "hazard_blocks_operation": False,
            "return_to_operational": {"trigger": TRIGGER_REPAIR_COMPLETE},
        },
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "ms",
            "hazard_blocks_operation": False,
            "return_to_operational": {"trigger": TRIGGER_REPAIR_COMPLETE},
        },
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "ls",
            "hazard_blocks_operation": False,
            "return_to_operational": {
                "trigger": TRIGGER_REPAIR_BELOW,
                "threshold": 2.0,
            },
        },
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "hospital",
            "hazard_blocks_operation": False,
            "return_to_operational": {"trigger": TRIGGER_REPAIR_COMPLETE},
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "clinic",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "doctors",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "apotheek",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "health",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        },
    ]
    return DependencyKnowledgeGraph.from_config(rules)
