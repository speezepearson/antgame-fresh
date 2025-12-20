"""Game mechanics and state."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from core import Timestamp, Pos, Region


# Game Constants
GRID_SIZE = 32
VISIBILITY_RADIUS = 8


class Team(Enum):
    RED = "RED"
    BLUE = "BLUE"


@dataclass
class MoveOrder:
    target: Pos
    then_return_home: bool = False


# What can be observed at a cell
@dataclass(frozen=True)
class Empty:
    pass


@dataclass(frozen=True)
class UnitPresent:
    team: Team


@dataclass(frozen=True)
class BasePresent:
    team: Team


CellContents = Empty | UnitPresent | BasePresent

# A logbook maps timestamps to observations (position -> list of contents)
Logbook = dict[Timestamp, dict[Pos, list[CellContents]]]


def get_base_region(team: Team) -> Region:
    """Get the base region for a team."""
    if team == Team.RED:
        center = Pos(2, GRID_SIZE // 2)
    else:
        center = Pos(GRID_SIZE - 3, GRID_SIZE // 2)

    # Create a 5x5 square centered on the position
    cells = frozenset(
        Pos(x, y)
        for x in range(center.x - 2, center.x + 3)
        for y in range(center.y - 2, center.y + 3)
    )
    return Region(cells)


@dataclass
class Unit:
    team: Team
    pos: Pos
    original_pos: Pos  # Where the unit spawned (for returning home)
    order: MoveOrder | None = None
    # Observations since last sync with home base
    logbook: Logbook = field(default_factory=dict)
    last_sync_tick: Timestamp = 0

    def home_base_region(self) -> Region:
        """Get this unit's home base region."""
        return get_base_region(self.team)

    def home_pos(self) -> Pos:
        """Get this unit's original spawn position (for returning home)."""
        return self.original_pos

    def is_near_base(self) -> bool:
        """Check if unit is inside its home base region."""
        return self.home_base_region().contains(self.pos)


@dataclass
class GameState:
    tick: Timestamp = 0
    units: list[Unit] = field(default_factory=list)
    selected_unit: Unit | None = None
    # Each team's home base logbook
    base_logbooks: dict[Team, Logbook] = field(default_factory=dict)
    # Slider positions for each team's view (which tick they're viewing)
    view_tick: dict[Team, Timestamp] = field(default_factory=dict)
    # Whether each player's view auto-advances to current tick
    view_live: dict[Team, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        red_region = get_base_region(Team.RED)
        blue_region = get_base_region(Team.BLUE)

        # Get center positions from regions for spawning
        red_cells = list(red_region.cells)
        blue_cells = list(blue_region.cells)
        red_center = Pos(2, GRID_SIZE // 2)
        blue_center = Pos(GRID_SIZE - 3, GRID_SIZE // 2)

        # Spawn 3 units inside each base
        for i in range(3):
            red_spawn = Pos(red_center.x, red_center.y - 1 + i)
            blue_spawn = Pos(blue_center.x, blue_center.y - 1 + i)
            self.units.append(Unit(Team.RED, red_spawn, red_spawn))
            self.units.append(Unit(Team.BLUE, blue_spawn, blue_spawn))

        self.base_logbooks[Team.RED] = {}
        self.base_logbooks[Team.BLUE] = {}
        self.view_tick[Team.RED] = 0
        self.view_tick[Team.BLUE] = 0
        self.view_live[Team.RED] = True
        self.view_live[Team.BLUE] = True

        # Record initial observations from base regions
        self._record_observations_from_bases()

    def get_base_region(self, team: Team) -> Region:
        """Get a team's base region."""
        return get_base_region(team)

    def get_base_pos(self, team: Team) -> Pos:
        """Get a position in the team's base (deprecated, for backward compatibility)."""
        # Return an arbitrary cell from the region
        return next(iter(self.get_base_region(team).cells))

    def _record_observations_from_bases(self) -> None:
        """Record observations from each base region's edge cells."""
        for team in [Team.RED, Team.BLUE]:
            observations = self._observe_from_base_region(team)
            if observations:
                if self.tick not in self.base_logbooks[team]:
                    self.base_logbooks[team][self.tick] = {}
                self.base_logbooks[team][self.tick].update(observations)

    def _observe_from_position(self, observer_pos: Pos) -> dict[Pos, list[CellContents]]:
        """Return what can be observed from a given position."""
        observations: dict[Pos, list[CellContents]] = {}

        # Check all positions within visibility radius
        for dx in range(-VISIBILITY_RADIUS, VISIBILITY_RADIUS + 1):
            for dy in range(-VISIBILITY_RADIUS, VISIBILITY_RADIUS + 1):
                if abs(dx) + abs(dy) <= VISIBILITY_RADIUS:
                    pos = Pos(observer_pos.x + dx, observer_pos.y + dy)
                    if 0 <= pos.x < GRID_SIZE and 0 <= pos.y < GRID_SIZE:
                        # Check what's at this position
                        contents = self._get_contents_at(pos)
                        observations[pos] = contents

        return observations

    def _observe_from_base_region(self, team: Team) -> dict[Pos, list[CellContents]]:
        """Observe from all edge cells of a base region, return union."""
        region = self.get_base_region(team)
        all_observations: dict[Pos, list[CellContents]] = {}
        for edge_pos in region.get_edge_cells():
            observations = self._observe_from_position(edge_pos)
            all_observations.update(observations)
        return all_observations

    def _get_contents_at(self, pos: Pos) -> list[CellContents]:
        """Determine what's actually at a position right now."""
        contents: list[CellContents] = []

        # Check for units
        for unit in self.units:
            if unit.pos == pos:
                contents.append(UnitPresent(unit.team))

        # Check for bases (any cell in the region)
        if self.get_base_region(Team.RED).contains(pos):
            contents.append(BasePresent(Team.RED))
        if self.get_base_region(Team.BLUE).contains(pos):
            contents.append(BasePresent(Team.BLUE))

        # If nothing found, return empty
        if not contents:
            contents.append(Empty())

        return contents

    def _merge_logbook_to_base(self, unit: Unit) -> None:
        """Merge a unit's logbook into the team's base logbook."""
        base_logbook = self.base_logbooks[unit.team]
        for timestamp, observations in unit.logbook.items():
            if timestamp not in base_logbook:
                base_logbook[timestamp] = {}
            # Merge observations for this timestamp
            base_logbook[timestamp].update(observations)

        # Clear unit's logbook and update sync time
        unit.logbook.clear()
        unit.last_sync_tick = self.tick


def tick_game(state: GameState) -> None:
    """Advance the game by one tick."""
    # Record observations for all units
    for unit in state.units:
        observations = state._observe_from_position(unit.pos)
        unit.logbook[state.tick] = observations

    # Move units
    for unit in state.units:
        if unit.order is None:
            continue

        target = unit.order.target
        dx = 0 if target.x == unit.pos.x else (1 if target.x > unit.pos.x else -1)
        dy = 0 if target.y == unit.pos.y else (1 if target.y > unit.pos.y else -1)

        if dx != 0:
            unit.pos = Pos(unit.pos.x + dx, unit.pos.y)
        elif dy != 0:
            unit.pos = Pos(unit.pos.x, unit.pos.y + dy)

        if unit.pos == target:
            if unit.order.then_return_home:
                unit.order = MoveOrder(target=unit.home_pos(), then_return_home=False)
            else:
                unit.order = None

    # Sync units that are at base
    for unit in state.units:
        if unit.is_near_base():
            state._merge_logbook_to_base(unit)

    state.tick += 1

    # Record observations from base regions for the new tick
    state._record_observations_from_bases()
