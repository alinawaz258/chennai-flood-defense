from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import networkx as nx

# Initialize Chennai street graph once at import time.
CITY_MAP = nx.Graph()
CITY_MAP.add_weighted_edges_from(
    [
        ("T_Nagar", "Guindy", 15),
        ("Guindy", "Velachery", 10),
        ("T_Nagar", "Saidapet", 8),
        ("Saidapet", "Velachery", 12),
    ]
)


def _build_safe_map(flooded_areas: Iterable[str]) -> nx.Graph:
    """Return a weighted graph with flooded nodes penalized for routing."""
    safe_map = CITY_MAP.copy()
    flood_penalty = 999

    for area in flooded_areas:
        if area in safe_map:
            for neighbor in safe_map.neighbors(area):
                safe_map[area][neighbor]["weight"] += flood_penalty

    return safe_map


def _heuristic(_: str, __: str) -> int:
    """A* heuristic kept at zero for guaranteed optimal routes on weighted graphs."""
    return 0


@lru_cache(maxsize=256)
def _cached_safe_route(start: str, destination: str, flooded_key: tuple[str, ...]) -> tuple[str, ...]:
    """Cache common path queries to avoid repeated shortest-path recomputation."""
    safe_map = _build_safe_map(flooded_key)
    route = nx.astar_path(
        safe_map,
        source=start,
        target=destination,
        heuristic=_heuristic,
        weight="weight",
    )
    return tuple(route)


def calculate_safe_route(start: str, destination: str, flooded_areas: list[str]) -> list[str]:
    """Compute a safe route while avoiding flooded areas when possible."""
    flooded_key = tuple(sorted(set(flooded_areas)))

    try:
        route = _cached_safe_route(start, destination, flooded_key)
        return list(route)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
