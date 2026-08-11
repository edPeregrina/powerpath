"""
Dependency Knowledge Graph for asset-hazard-asset relationships.

Defines per-pair dependency rules between hazard types, primary asset types (A),
and optionally downstream asset types (B) within A's service area.

Usage example::

    from src.dependency_knowledge_graph import DependencyKnowledgeGraph

    kg = DependencyKnowledgeGraph.from_config([
        {
            'hazard_type': 'flooding',
            'asset_type_a': 'msls',
            'asset_type_b': None,
            'relationship': 'direct',
            'parameters': {
                'hazard_blocks_operation': True,
                'return_to_operational': {'trigger': 'repair_complete'},
            },
        },
    ])

    rules = kg.get_rules('flooding', 'msls')
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Return-to-operational trigger constants
# ---------------------------------------------------------------------------
TRIGGER_IMMEDIATE = "immediate"
"""Asset returns to operational as soon as hazard clears (no repair required)."""

TRIGGER_REPAIR_COMPLETE = "repair_complete"
"""Asset returns to operational only after repair_time reaches 0."""

TRIGGER_REPAIR_BELOW = "repair_below"
"""Asset returns to operational when repair_time < threshold (e.g. 2.0 hours)."""

VALID_TRIGGERS = {TRIGGER_IMMEDIATE, TRIGGER_REPAIR_COMPLETE, TRIGGER_REPAIR_BELOW}

# ---------------------------------------------------------------------------
# Default rule parameters (used when no matching rule is found in the graph)
# ---------------------------------------------------------------------------
_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "hazard_blocks_operation": True,
    "return_to_operational": {"trigger": TRIGGER_IMMEDIATE},
}


@dataclass
class ReturnToOperational:
    """Describes what must happen before an asset can return to operational state.

    Attributes:
        trigger: One of ``"immediate"``, ``"repair_complete"``, or ``"repair_below"``.
        threshold: Only used when *trigger* is ``"repair_below"``; the asset
            becomes operational once ``repair_time < threshold``.
    """

    trigger: str = TRIGGER_IMMEDIATE
    threshold: float = 0.0

    def __post_init__(self):
        if self.trigger not in VALID_TRIGGERS:
            raise ValueError(
                f"Invalid trigger '{self.trigger}'. Must be one of {VALID_TRIGGERS}."
            )
        if self.trigger == TRIGGER_REPAIR_BELOW and self.threshold <= 0.0:
            raise ValueError(
                "threshold must be > 0 when trigger is 'repair_below'."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReturnToOperational":
        return cls(
            trigger=d.get("trigger", TRIGGER_IMMEDIATE),
            threshold=float(d.get("threshold", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"trigger": self.trigger}
        if self.trigger == TRIGGER_REPAIR_BELOW:
            result["threshold"] = self.threshold
        return result


@dataclass
class DependencyRule:
    """A single dependency rule between a hazard type and one or two asset types.

    Attributes:
        hazard_type: The hazard type this rule applies to (e.g. ``"flooding"``).
        asset_type_a: The primary asset type (e.g. ``"msls"``).
        asset_type_b: Optional downstream asset type within A's service area
            (e.g. ``"hospital"``).  ``None`` means the rule applies to A itself.
        relationship: ``"direct"`` (rule on A) or ``"service_area"`` (A → B).
        hazard_blocks_operation: If ``True``, assets are non-operational while
            hazard exposure exceeds the simulation flood threshold.
        return_to_operational: Specifies what triggers the return to operational.
    """

    hazard_type: str
    asset_type_a: str
    asset_type_b: Optional[str]
    relationship: str
    hazard_blocks_operation: bool
    return_to_operational: ReturnToOperational

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DependencyRule":
        params = d.get("parameters", {})
        rto_dict = params.get("return_to_operational", {"trigger": TRIGGER_IMMEDIATE})
        return cls(
            hazard_type=d["hazard_type"],
            asset_type_a=d["asset_type_a"],
            asset_type_b=d.get("asset_type_b"),
            relationship=d.get("relationship", "direct"),
            hazard_blocks_operation=bool(params.get("hazard_blocks_operation", True)),
            return_to_operational=ReturnToOperational.from_dict(rto_dict),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_type": self.hazard_type,
            "asset_type_a": self.asset_type_a,
            "asset_type_b": self.asset_type_b,
            "relationship": self.relationship,
            "parameters": {
                "hazard_blocks_operation": self.hazard_blocks_operation,
                "return_to_operational": self.return_to_operational.to_dict(),
            },
        }


class DependencyKnowledgeGraph:
    """Container for all dependency rules used in a simulation.

    Rules are keyed by ``(hazard_type, asset_type_a, asset_type_b)`` for fast
    look-up.  When no rule is registered for a given combination, a safe
    default rule is returned (hazard blocks operation; trigger = immediate).

    Args:
        rules: Initial list of :class:`DependencyRule` objects.
    """

    def __init__(self, rules: Optional[List[DependencyRule]] = None):
        self._rules: List[DependencyRule] = rules or []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, rule_list: List[Dict[str, Any]]) -> "DependencyKnowledgeGraph":
        """Build a graph from a list of rule dicts (e.g. from ``config.py``)."""
        rules = [DependencyRule.from_dict(r) for r in rule_list]
        return cls(rules)

    @classmethod
    def from_json(cls, json_str: str) -> "DependencyKnowledgeGraph":
        """Deserialise from a JSON string."""
        return cls.from_config(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        """Serialise all rules to a JSON string."""
        return json.dumps([r.to_dict() for r in self._rules], indent=indent)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_rules(
        self,
        hazard_type: str,
        asset_type_a: str,
        asset_type_b: Optional[str] = None,
    ) -> List[DependencyRule]:
        """Return rules matching the given combination.

        Args:
            hazard_type: The active hazard type (e.g. ``"flooding"``).
            asset_type_a: The primary asset type.
            asset_type_b: Optional downstream asset type.  Pass ``None`` to
                retrieve direct rules only.

        Returns:
            List of matching :class:`DependencyRule` objects (may be empty).
        """
        return [
            r
            for r in self._rules
            if r.hazard_type == hazard_type
            and r.asset_type_a == asset_type_a
            and r.asset_type_b == asset_type_b
        ]

    def get_default_rule(self, hazard_type: str, asset_type_a: str) -> DependencyRule:
        """Return a safe default rule for pairs not present in the graph."""
        return DependencyRule(
            hazard_type=hazard_type,
            asset_type_a=asset_type_a,
            asset_type_b=None,
            relationship="direct",
            hazard_blocks_operation=True,
            return_to_operational=ReturnToOperational(trigger=TRIGGER_IMMEDIATE),
        )

    def get_rules_or_default(
        self,
        hazard_type: str,
        asset_type_a: str,
        asset_type_b: Optional[str] = None,
    ) -> List[DependencyRule]:
        """Like :meth:`get_rules` but returns the default rule when none match."""
        matched = self.get_rules(hazard_type, asset_type_a, asset_type_b)
        if matched:
            return matched
        return [self.get_default_rule(hazard_type, asset_type_a)]

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def all_asset_types(self) -> List[str]:
        """Return the sorted unique set of all asset_type_a values."""
        return sorted({r.asset_type_a for r in self._rules})

    def all_hazard_types(self) -> List[str]:
        """Return the sorted unique set of all hazard_type values."""
        return sorted({r.hazard_type for r in self._rules})

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"DependencyKnowledgeGraph({len(self._rules)} rules)"


# ---------------------------------------------------------------------------
# Default knowledge graph for existing electricity infrastructure assets
# ---------------------------------------------------------------------------

def build_default_knowledge_graph() -> DependencyKnowledgeGraph:
    """Return the baseline knowledge graph for flooding and the known asset types.

    Asset types currently in the model:

    * ``"msls"`` – Medium/Low-voltage Substation (MV/LV, combined).  These are
      the primary distribution substations.  Following NKWK the median failure
      depth is 0.6 m; they require physical repair before returning to service.
    * ``"ms"`` – Medium-voltage Substation (MV only).  Similar criticality to
      msls; assumed to require repair before returning to service.
    * ``"ls"`` – Low-voltage Substation (LV only).  Smaller and simpler;
      assumed to return to service once repair time has been reduced below a
      threshold (default 2 hours) rather than waiting for full completion.

    All rules use ``hazard_type = "flooding"`` — the only hazard currently
    modelled.  Add further calls to ``DependencyKnowledgeGraph.from_config``
    (or extend this list) to introduce new hazards (wind, seismic, …) or new
    asset types (hospitals, water-treatment, …).

    The rules encoded here represent the **defaults when nothing is explicitly
    parameterised** and serve as a living template for future extensions.

    Returns:
        A :class:`DependencyKnowledgeGraph` populated with sensible defaults.
    """
    rules = [
        # ------------------------------------------------------------------
        # msls – Medium/Low-voltage Substation
        # ------------------------------------------------------------------
        # While flooded the substation is non-operational.
        # It can only return to service once full repair is complete
        # (repair_time == 0), reflecting the higher structural complexity and
        # safety requirements of combined MV/LV stations.
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": None,
            "relationship": "direct",
            "parameters": {
                "hazard_blocks_operation": True,
                "return_to_operational": {
                    "trigger": TRIGGER_REPAIR_COMPLETE,
                },
            },
        },
        # ------------------------------------------------------------------
        # ms – Medium-voltage Substation
        # ------------------------------------------------------------------
        # Same behaviour as msls: requires full repair before returning to
        # service.  Medium-voltage equipment typically needs certified
        # inspection after flood exposure.
        {
            "hazard_type": "flooding",
            "asset_type_a": "ms",
            "asset_type_b": None,
            "relationship": "direct",
            "parameters": {
                "hazard_blocks_operation": True,
                "return_to_operational": {
                    "trigger": TRIGGER_REPAIR_COMPLETE,
                },
            },
        },
        # ------------------------------------------------------------------
        # ls – Low-voltage Substation
        # ------------------------------------------------------------------
        # Flooded ls stations are taken out of service, but they are simpler
        # units that can be re-energised once the remaining repair work has
        # dropped below 2 hours (i.e. a quick inspection / dry-out is
        # sufficient rather than a full replacement).
        {
            "hazard_type": "flooding",
            "asset_type_a": "ls",
            "asset_type_b": None,
            "relationship": "direct",
            "parameters": {
                "hazard_blocks_operation": True,
                "return_to_operational": {
                    "trigger": TRIGGER_REPAIR_BELOW,
                    "threshold": 2.0,   # hours – matches global repair_threshold default
                },
            },
        },
    ]
    return DependencyKnowledgeGraph.from_config(rules)
