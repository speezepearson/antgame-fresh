"""Game mechanics and state."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from core import Timestamp, Pos, Region
from logbook import Logbook
from mechanics import (
    CellContents,
    Empty,
    GameState,
    PlayerAction,
    Team,
    Unit,
    UnitId,
    UnitPresent,
)
from planning import PlanningMind

LastObservations = dict[Pos, tuple[Timestamp, list[CellContents]]]


@dataclass
class PlayerKnowledge:
    team: Team
    grid_width: int
    grid_height: int
    tick: Timestamp

    logbook: Logbook = field(default_factory=Logbook)
    last_in_base: dict[UnitId, tuple[Timestamp, Unit]] = field(default_factory=dict)
    expected_trajectories: dict[UnitId, ExpectedTrajectory] = field(
        default_factory=dict
    )
    own_units_in_base: dict[UnitId, Unit] = field(default_factory=dict)

    # def add_raw_observations(
    #     self, game: GameState, observations: RawObservations
    # ) -> None:
    #     self.all_observations.setdefault(game.now, {}).update(observations)
    #     for pos, contents_list in observations.items():
    #         self.last_observations[pos] = (game.now, contents_list)
    #         for contents in contents_list:
    #             if isinstance(contents, UnitPresent):
    #                 self.last_seen[contents.unit_id] = (
    #                     game.now,
    #                     game.units[contents.unit_id],
    #                 )

    # def record_observations_from_bases(self, game: GameState) -> None:
    #     """Record observations from each base region and units in the base."""
    #     observations = self.observe_from_base_region(game)

    #     # Add observations from units currently in the base
    #     # TODO: can we just skip this since we'll immediately siphon their logs anyway?
    #     for unit in game.units.values():
    #         if unit.team == self.team and unit.is_in_base(game):
    #             unit_observations = game.observe_from_position(
    #                 unit.pos, unit.visibility_radius
    #             )
    #             observations.update(unit_observations)

    #     self.add_raw_observations(game, observations)

    # def observe_from_base_region(
    #     self, game: GameState
    # ) -> dict[Pos, list[CellContents]]:
    #     """Observe only the tiles within the base region itself."""
    #     region = game.get_base_region(self.team)
    #     observations: dict[Pos, list[CellContents]] = {}
    #     for pos in region.cells:
    #         contents = game.get_contents_at(pos)
    #         observations[pos] = contents
    #     return observations

    # def get_currently_visible_cells(self) -> set[Pos]:
    #     """Get all cells currently visible to the player (from their observations at current tick)."""
    #     if not self.logbook.observation_log: return set()
    #     last_tick = max(self.logbook.observation_log.keys())
    #     return set(self.logbook.observation_log.get(last_tick, {}).keys())

    def observe(self, tick: Timestamp, own_units_in_base: list[Unit]) -> None: # previous named `tick_knowledge`
        # if self.team == Team.RED:
        #     print('unit posns:', {unit.id: unit.pos for unit in own_units_in_base})
        #     print('observing at', tick, 'own units in base:', [unit.id for unit in own_units_in_base], 'seeing', {
        #         unit.id: [cast(PlanningMind, unit.mind).logbook._latest_observations_at, {k:v for k,v in cast(PlanningMind, unit.mind).logbook.latest_observations.items() if v != [Empty()]}]
        #         for unit in own_units_in_base
        #     })
        self.logbook.add_latest_observations(tick, {})  # we want the logbook to know it's stale if no observations come in
        self.own_units_in_base = {unit.id: unit for unit in own_units_in_base}
        self.last_in_base.update({unit.id: (self.tick, deepcopy(unit)) for unit in own_units_in_base})
        # if self.team == Team.RED: print('units_in_base:', units_in_base)
        for unit in own_units_in_base:
            unit_logbook = cast(PlanningMind, unit.mind).logbook
            self.logbook.copy_from(unit_logbook)
            unit_logbook.clear()

        self.compute_expected_trajectories()
        # if tick > 30:
        #     breakpoint()
        # self.record_observations_from_bases(state)
        self.tick = tick

    def compute_expected_trajectories(self) -> None:
        visible_units = {
            cc.unit_id
            for ccs in self.logbook.latest_observations.values()
            for cc in ccs
            if isinstance(cc, UnitPresent)}
        # Compute last known trajectories for units not currently visible
        for unit_id, (last_seen_tick, last_seen_unit) in self.last_in_base.items():
            # print('latest', self.logbook.latest_observations)
            if unit_id in visible_units:
                # if unit_id in self.expected_trajectories:
                #     print('DEBUG: clearing trajectory for unit', unit_id)
                # Unit is visible - clear any trajectory
                self.expected_trajectories.pop(unit_id, None)
            elif unit_id not in self.expected_trajectories:
                self.expected_trajectories[unit_id] = compute_expected_trajectory(
                    last_seen_unit, self, start_tick=last_seen_tick
                )


ExpectedTrajectory = dict[Timestamp, Pos]


def compute_expected_trajectory(
    unit: Unit,
    knowledge: PlayerKnowledge,
    start_tick: Timestamp,
    max_ticks: int = 100,
) -> ExpectedTrajectory:
    """Simulate unit's plan in isolation to predict its path.

    Creates a minimal game state with just this unit (no food, no other units)
    and runs tick_game to simulate movement including interrupt checks.
    """
    # Create minimal game state with just this unit
    sim_state = GameState(
        grid_width=knowledge.grid_width,
        grid_height=knowledge.grid_height,
        base_regions={
            team: Region(frozenset([Pos(0, 0)])) for team in Team
        },  # TODO: hack
        now=start_tick,
        units={unit.id: deepcopy(unit)},
        food={},  # No food in simulation
    )

    trajectory = {start_tick: unit.pos}
    sim_unit = sim_state.units[unit.id]

    for _ in range(max_ticks):
        sim_state.tick()
        trajectory[sim_state.now] = sim_unit.pos

    return trajectory
