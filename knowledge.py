"""Game mechanics and state."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeVar, Generic, Callable, Any, NewType
import random
import numpy as np

from core import Timestamp, Pos, Region
from mechanics import BasePresent, CellContents, Empty, FoodPresent, GameState, ObservationLog, RawObservations, Team, Unit, UnitId, UnitPresent, tick_game
from perlin import perlin

LastObservations = dict[Pos, tuple[Timestamp, list[CellContents]]]

@dataclass
class PlayerKnowledge:
    team: Team
    grid_width: int
    grid_height: int
    tick: Timestamp
    own_units_in_base: list[Unit]

    all_observations: ObservationLog = field(default_factory=dict)
    last_seen: dict[UnitId, tuple[Timestamp, Unit]] = field(default_factory=dict)
    expected_trajectories: dict[UnitId, ExpectedTrajectory] = field(default_factory=dict)
    last_observations: LastObservations = field(default_factory=dict)

    def merge_observation_log(self, log: ObservationLog) -> None:
        """Merge a unit's observations into the player's knowledge."""
        for timestamp, raw_observations in log.items():
            self.all_observations.setdefault(timestamp, {}).update(raw_observations)
            for pos, contents_list in raw_observations.items():
                if not (pos in self.last_observations and self.last_observations[pos][0] >= timestamp):
                    self.last_observations[pos] = (timestamp, contents_list)

    def siphon_unit_logs(self) -> None:
        for unit in self.own_units_in_base:
            self.merge_observation_log(unit.observation_log)
            # Clear unit's logbook and update sync time
            unit.observation_log.clear()
            unit.last_sync_tick = self.tick


    def get_currently_visible_cells(self) -> set[Pos]:
        """Get all cells currently visible to the player (from their observations at current tick)."""
        last_tick = max(self.all_observations.keys())
        return set(self.all_observations.get(last_tick, {}).keys())


    def tick_knowledge(self, tick: Timestamp, new_units_in_base: list[Unit]) -> None:
        self.tick = tick
        self.own_units_in_base = new_units_in_base
        self.last_seen.update({unit.id: (tick, unit) for unit in new_units_in_base})
        self.siphon_unit_logs()
        self.compute_expected_trajectories()

    def compute_expected_trajectories(self) -> None:
        # currently_visible_unit_ids = {
        #     content.unit_id
        #     for contents in self.all_observations.get(self.tick, {}).values()
        #     for content in contents
        #     if isinstance(content, UnitPresent)
        # }
        currently_visible_unit_ids = {unit.id for unit in self.own_units_in_base}

        # Compute last known trajectories for units not currently visible
        for tick, unit in self.last_seen.values():
            if unit.id in currently_visible_unit_ids:
                self.expected_trajectories.pop(unit.id, None)
            elif unit.id in self.last_seen and unit.id not in self.expected_trajectories:
                last_seen_tick, last_seen_unit = self.last_seen[unit.id]

                sim_state = GameState(
                    grid_width=self.grid_width,
                    grid_height=self.grid_height,
                    base_regions={team: Region(frozenset([Pos(0,0)])) for team in Team}, # TODO: hack
                    tick=last_seen_tick,
                    units=[deepcopy(unit)],
                    food={},  # No food in simulation
                )

                positions = [unit.pos]
                sim_unit = sim_state.units[0]

                for _ in range(100):
                    if not sim_unit.plan.orders:
                        break

                    tick_game(sim_state)
                    positions.append(sim_unit.pos)

                self.expected_trajectories[unit.id] = ExpectedTrajectory(
                    unit_id=unit.id,
                    start_tick=self.tick,
                    positions=positions,
                )


@dataclass
class ExpectedTrajectory:
    """Predicted path of a unit that has left the visible area."""

    unit_id: UnitId
    start_tick: Timestamp  # When the trajectory was computed
    positions: list[Pos]  # positions[i] = where unit should be at start_tick + i


def compute_expected_trajectory(
    unit: Unit,
    state: GameState,
    start_tick: Timestamp,
    max_ticks: int = 100,
) -> ExpectedTrajectory:
    """Simulate unit's plan in isolation to predict its path.

    Creates a minimal game state with just this unit (no food, no other units)
    and runs tick_game to simulate movement including interrupt checks.
    """
    # Create minimal game state with just this unit
    sim_state = GameState(
        grid_width=state.grid_width,
        grid_height=state.grid_height,
        base_regions={team: Region(frozenset([Pos(0,0)])) for team in Team}, # TODO: hack
        tick=start_tick,
        units=[deepcopy(unit)],
        food={},  # No food in simulation
    )

    positions = [unit.pos]
    sim_unit = sim_state.units[0]

    for _ in range(max_ticks):
        if not sim_unit.plan.orders:
            break

        tick_game(sim_state)
        positions.append(sim_unit.pos)

    return ExpectedTrajectory(
        unit_id=unit.id,
        start_tick=state.tick,
        positions=positions,
    )
