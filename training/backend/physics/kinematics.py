"""Four-branch landslide velocity solver and H3 Voellmy-style routing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h3
import numpy as np

from .multi_causal_engine import PhysicsInputs, PhysicsState

GRAVITY_M_S2 = 9.81


@dataclass
class KinematicsState:
    peak_velocity_m_s: np.ndarray
    velocity_class: np.ndarray
    velocity_class_name: np.ndarray
    failure_mode: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "peak_velocity_m_s": self.peak_velocity_m_s.tolist(),
            "velocity_class": self.velocity_class.tolist(),
            "velocity_class_name": self.velocity_class_name.tolist(),
            "failure_mode": self.failure_mode.tolist(),
        }


@dataclass
class RunoutRoute:
    source_cell: str
    path: List[str]
    arrival_time_min: float
    final_energy_head_m: float
    intersected_roads: int
    intersected_buildings: int
    intersected_households: Optional[int]
    stopped_at: str
    peak_velocity_m_s: float = 0.
    stop_reason: str = 'energy_exhausted_or_local_sink'

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def solve_kinematics(inputs: PhysicsInputs, state: PhysicsState) -> KinematicsState:
    """Report mechanical mode; velocity requires an explicit travel-length scenario."""
    slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
    burn = state.effective_burn_severity
    rain = np.maximum(np.asarray(inputs.rain_24h_mm, dtype=np.float64), 0.0)
    snow_depth = np.maximum(np.asarray(inputs.snow_depth_m, dtype=np.float64), 0.0)
    pga = np.maximum(np.asarray(inputs.pga_g, dtype=np.float64), 0.0)
    tan_phi = np.tan(np.radians(state.friction_angle_deg))

    applicable = state.newmark_applicable if state.newmark_applicable is not None else state.fs_static > 1.
    coseismic = applicable & (pga > state.critical_acceleration_g) & (state.newmark_displacement_cm > 0)
    rapid = np.zeros_like(coseismic)  # basin runoff probability does not identify flow velocity
    slump = state.fs_dynamic < 1.
    stable = ~(coseismic | slump)
    length = np.broadcast_to(np.asarray(inputs.runout_travel_length_m, float), state.fs_dynamic.shape)
    # Optional Coulomb-block energy scenario with supplied travel length. Without
    # geometry, a failure/displacement score cannot determine a velocity.
    acceleration = GRAVITY_M_S2*np.maximum(0., np.sin(slope)+state.kh*np.cos(slope)-tan_phi*np.cos(slope))
    speed = np.where(~stable & (length > 0), np.sqrt(2*acceleration*np.maximum(length, 0)), 0.)
    velocity_class = np.select(
        [
            speed >= 5.0,
            speed >= 5e-2,
            speed >= 5e-4,
            speed >= 5e-6,
            speed >= 5e-8,
            speed >= 5e-10,
        ],
        [7, 6, 5, 4, 3, 2],
        default=0,
    ).astype(np.int8)
    class_names = np.asarray(
        [
            "Stable",
            "Extremely Slow",
            "Very Slow",
            "Slow",
            "Moderate",
            "Rapid",
            "Very Rapid",
            "Extremely Rapid",
        ],
        dtype=object,
    )[velocity_class]
    class_names = np.where((~stable) & (length <= 0), "Unresolved: travel geometry unavailable", class_names)
    modes = np.select(
        [rapid, coseismic, slump, stable],
        ["Unresolved flow", "Newmark sliding-block displacement", "Mechanical instability / unresolved motion", "No modeled mechanical instability"],
        default="Unclassified",
    )
    return KinematicsState(speed, velocity_class, class_names, modes)


def select_runout_hotspots(state: PhysicsState, probability: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices meeting the defined FS or probability hotspot thresholds."""
    selected = state.fs_dynamic <= 1.05
    if probability is not None:
        probability = np.asarray(probability, dtype=np.float64)
        if probability.shape != state.fs_dynamic.shape:
            raise ValueError("probability must have the same shape as PhysicsState")
        selected = selected | (probability >= 0.65)
    return np.where(selected)[0]


def route_runout(
    source_cell: str,
    terrain_cells: Mapping[str, object],
    peak_velocity_m_s: float,
    mu_b: float,
    xi: float,
    infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
    delta_s_by_edge: Optional[Mapping[Tuple[str, str], float]] = None,
    max_steps: int = 50,
    flow_depth_m: float = 0.5,
    released: bool = False,
) -> RunoutRoute:
    """Route positive energy head over lower H3 neighbors.

    Edge length is in metres and xi in m/s². Integrate the constant-edge law
    dE/ds = sin(theta)-mu*cos(theta)-2gE/(xi*h) exactly. ``released`` authorizes
    acceleration from rest; it does not assert that release actually occurred.
    """
    if source_cell not in terrain_cells:
        raise KeyError("source cell is absent from the terrain graph")
    if not math.isfinite(mu_b) or not math.isfinite(xi) or mu_b < 0 or xi <= 0 or max_steps < 1:
        raise ValueError("mu_b must be nonnegative, xi positive, and max_steps at least 1")
    if peak_velocity_m_s < 0 or not math.isfinite(peak_velocity_m_s):
        raise ValueError("peak velocity must be finite and nonnegative")

    if not math.isfinite(flow_depth_m) or flow_depth_m <= 0:
        raise ValueError("Flow depth must be positive and finite")
    h_f = float(flow_depth_m)
    energy_head = peak_velocity_m_s**2 / (2.0 * GRAVITY_M_S2)
    path = [source_cell]
    elapsed_seconds = 0.0
    current = source_cell
    visited = {source_cell}
    infra = infrastructure_by_cell or {}
    edge_s = delta_s_by_edge or {}
    peak_speed=peak_velocity_m_s
    stop_reason='energy_exhausted_or_local_sink'

    for _ in range(min(int(max_steps), 50)):
        if energy_head <= 0.0 and not released:
            break
        cell = terrain_cells[current]
        neighbors = list(getattr(cell, "downslope_neighbors", []) or [])
        candidates = []
        current_elevation = float(getattr(cell, "elevation_m"))
        current_speed = math.sqrt(max(0.0, 2.0 * GRAVITY_M_S2 * energy_head))
        for neighbor_id in neighbors:
            if neighbor_id in visited or neighbor_id not in terrain_cells:
                continue
            neighbor = terrain_cells[neighbor_id]
            dz = current_elevation - float(getattr(neighbor, "elevation_m"))
            if dz <= 0.0:
                continue
            dx = _distance_m(cell, neighbor)
            delta_s = float(edge_s.get((current,neighbor_id),math.hypot(dx,dz)))
            if not math.isfinite(delta_s) or delta_s<=0:raise ValueError('Edge travel length must be positive')
            # Exact constant-edge solution of dE/ds=sin(theta)-mu*cos(theta)
            # -2gE/(xi*h), xi in m/s^2. Stable for coarse H3 edges, including
            # release from rest on a positive net-driving slope.
            driving=(dz-mu_b*dx)/delta_s
            decay=2*GRAVITY_M_S2/(xi*h_f)
            equilibrium=driving/decay
            next_head=equilibrium+(energy_head-equilibrium)*math.exp(-decay*delta_s)
            if next_head > 0.0:
                candidates.append((next_head, neighbor_id, delta_s))
        if not candidates:
            stop_reason='no_traversable_downstream_edge'
            break
        next_head, next_cell, dx = max(candidates, key=lambda candidate: (candidate[0],candidate[1]))
        next_speed = math.sqrt(2.0 * GRAVITY_M_S2 * next_head)
        from scipy.integrate import quad
        next_elevation=float(getattr(terrain_cells[next_cell],'elevation_m'))
        horizontal=_distance_m(cell,terrain_cells[next_cell])
        drive=(current_elevation-next_elevation-mu_b*horizontal)/dx
        rate=2*GRAVITY_M_S2/(xi*h_f);eq=drive/rate;start_head=energy_head
        def inverse_speed(distance):
            e=start_head*math.exp(-rate*distance)+eq*(-math.expm1(-rate*distance))
            return 1/math.sqrt(max(2*GRAVITY_M_S2*e,np.finfo(float).tiny))
        elapsed_seconds += quad(inverse_speed,0.,dx,epsabs=1e-7,limit=100)[0]
        energy_head = next_head
        peak_speed=max(peak_speed,next_speed)
        current = next_cell
        visited.add(current)
        path.append(current)

    def unique_assets(category):
        identifiers=set();legacy=0
        for cell_id in path:
            record=infra.get(cell_id,{})
            if category+'_ids' in record:identifiers.update(record[category+'_ids'])
            else:legacy+=int(record.get(category,0))
        return len(identifiers)+legacy
    road_hits = unique_assets('roads')
    building_hits = unique_assets('buildings')
    household_counts = [infra.get(cell_id, {}).get("households") for cell_id in path]
    household_hits = (
        sum(int(value) for value in household_counts)
        if all(value is not None for value in household_counts)
        else None
    )
    if len(path)>=min(int(max_steps),50)+1:stop_reason='step_limit_domain_truncated'
    return RunoutRoute(
        source_cell=source_cell,
        path=path,
        arrival_time_min=elapsed_seconds / 60.0,
        final_energy_head_m=energy_head,
        intersected_roads=road_hits,
        intersected_buildings=building_hits,
        intersected_households=household_hits,
        stopped_at=path[-1],
        peak_velocity_m_s=peak_speed,stop_reason=stop_reason,
    )


def route_hotspots(
    terrain_cells: Mapping[str, object],
    cell_ids: Sequence[str],
    state: PhysicsState,
    kinematics: KinematicsState,
    mu_b: float,
    xi: float,
    probability: Optional[np.ndarray] = None,
    infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
    max_routes: int = 10,
) -> List[RunoutRoute]:
    """Route up to ``max_routes`` cells meeting FS≤1.05 or failure probability≥0.65."""
    if len(cell_ids) != state.cell_count or kinematics.peak_velocity_m_s.size != state.cell_count:
        raise ValueError("cell_ids, PhysicsState, and KinematicsState must have matching cell counts")
    if max_routes < 1:
        return []
    hotspots = select_runout_hotspots(state, probability)
    if probability is not None:
        p = np.asarray(probability, dtype=np.float64)
        hotspots = sorted(hotspots, key=lambda idx: (p[idx], -state.fs_dynamic[idx]), reverse=True)
    else:
        hotspots = sorted(hotspots, key=lambda idx: (state.p_failure_physical[idx], -state.fs_dynamic[idx]), reverse=True)
    routes = []
    for idx in hotspots[:max_routes]:
        cell_id = cell_ids[int(idx)]
        if cell_id not in terrain_cells:
            continue
        routes.append(
            route_runout(
                cell_id,
                terrain_cells,
                float(kinematics.peak_velocity_m_s[idx]),
                mu_b,
                xi,
                infrastructure_by_cell=infrastructure_by_cell,
            )
        )
    return routes


def _distance_m(a: object, b: object) -> float:
    a_lat, a_lon = float(getattr(a, "latitude")), float(getattr(a, "longitude"))
    b_lat, b_lon = float(getattr(b, "latitude")), float(getattr(b, "longitude"))
    try:
        return float(h3.great_circle_distance((a_lat, a_lon), (b_lat, b_lon), unit="m"))
    except TypeError:
        lat1, lat2 = math.radians(a_lat), math.radians(b_lat)
        dlat, dlon = lat2 - lat1, math.radians(b_lon - a_lon)
        hav = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        return 2.0 * 6371008.8 * math.asin(math.sqrt(hav))
