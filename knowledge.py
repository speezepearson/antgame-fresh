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
from mechanics import (
    BasePresent,
    CellContents,
    Empty,
    FoodPresent,
    GameState,
    ObservationLog,
    RawObservations,
    Team,
    Unit,
    UnitId,
    UnitPresent,
    tick_game,
)
from perlin import perlin

LastObservations = dict[Pos, tuple[Timestamp, list[CellContents]]]


@dataclass
class PlayerKnowledge:
    team: Team
    grid_width: int
    grid_height: int
    tick: Timestamp

    all_observations: ObservationLog = field(default_factory=dict)
    last_seen: dict[UnitId, tuple[Timestamp, Unit]] = field(default_factory=dict)
    expected_trajectories: dict[UnitId, ExpectedTrajectory] = field(
        default_factory=dict
    )
    last_observations: LastObservations = field(default_factory=dict)

    def add_raw_observations(
        self, game: GameState, observations: RawObservations
    ) -> None:
        self.all_observations.setdefault(game.tick, {}).update(observations)
        for pos, contents_list in observations.items():
            self.last_observations[pos] = (game.tick, contents_list)
            for contents in contents_list:
                if isinstance(contents, UnitPresent):
                    self.last_seen[contents.unit_id] = (
                        game.tick,
                        game.units[contents.unit_id],
                    )

    def merge_observation_log(self, log: ObservationLog) -> None:
        """Merge a unit's observations into the player's knowledge."""
        for timestamp, raw_observations in log.items():
            self.all_observations.setdefault(timestamp, {}).update(raw_observations)
            for pos, contents_list in raw_observations.items():
                if not (
                    pos in self.last_observations
                    and self.last_observations[pos][0] >= timestamp
                ):
                    self.last_observations[pos] = (timestamp, contents_list)

    def record_observations_from_bases(self, game: GameState) -> None:
        """Record observations from each base region and units in the base."""
        observations = self.observe_from_base_region(game)

        # Add observations from units currently in the base
        # TODO: can we just skip this since we'll immediately siphon their logs anyway?
        for unit in game.units.values():
            if unit.team == self.team and unit.is_in_base(game):
                unit_observations = game.observe_from_position(
                    unit.pos, unit.visibility_radius
                )
                observations.update(unit_observations)

        self.add_raw_observations(game, observations)

    def observe_from_base_region(
        self, game: GameState
    ) -> dict[Pos, list[CellContents]]:
        """Observe only the tiles within the base region itself."""
        region = game.get_base_region(self.team)
        observations: dict[Pos, list[CellContents]] = {}
        for pos in region.cells:
            contents = game.get_contents_at(pos)
            observations[pos] = contents
        return observations

    def siphon_unit_logs(self, game: GameState) -> None:
        for unit in game.units.values():
            if unit.team == self.team and unit.is_in_base(game):
                self.merge_observation_log(unit.observation_log)
                # Clear unit's logbook and update sync time
                unit.observation_log.clear()
                unit.last_sync_tick = game.tick

    def get_currently_visible_cells(self) -> set[Pos]:
        """Get all cells currently visible to the player (from their observations at current tick)."""
        last_tick = max(self.all_observations.keys())
        return set(self.all_observations.get(last_tick, {}).keys())

    def tick_knowledge(self, state: GameState) -> None:
        self.siphon_unit_logs(state)
        self.compute_expected_trajectories(state)
        self.record_observations_from_bases(state)
        self.tick = state.tick

    def compute_expected_trajectories(self, state: GameState) -> None:
        visible = self.get_currently_visible_cells()

        # Compute last known trajectories for units not currently visible
        for unit in state.units.values():
            if unit.team != self.team:
                continue

            if unit.pos in visible:
                # Unit is visible - clear any trajectory
                self.expected_trajectories.pop(unit.id, None)
            elif (
                unit.id in self.last_seen and unit.id not in self.expected_trajectories
            ):
                last_seen_tick, last_seen_unit = self.last_seen[unit.id]
                self.expected_trajectories[unit.id] = compute_expected_trajectory(
                    last_seen_unit, state, start_tick=last_seen_tick
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
        base_regions={
            team: Region(frozenset([Pos(0, 0)])) for team in Team
        },  # TODO: hack
        tick=start_tick,
        units={unit.id: deepcopy(unit)},
        food={},  # No food in simulation
    )

    positions = [unit.pos]
    sim_unit = sim_state.units[unit.id]

    for _ in range(max_ticks):
        tick_game(sim_state)
        positions.append(sim_unit.pos)

    return ExpectedTrajectory(
        unit_id=unit.id,
        start_tick=state.tick,
        positions=positions,
    )
