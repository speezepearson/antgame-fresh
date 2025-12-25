"""Game mechanics and state."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol, TypeVar, Generic, Callable, Any, NewType
import random
import numpy as np

from core import Timestamp, Pos, Region
from perlin import perlin


# Game Constants
GRID_SIZE = 32

# Unit identification
UnitId = NewType("UnitId", int)
_next_unit_id: int = 0


def generate_unit_id() -> UnitId:
    """Generate a unique unit ID."""
    global _next_unit_id
    uid = UnitId(_next_unit_id)
    _next_unit_id += 1
    return uid


@dataclass
class FoodConfig:
    """Configuration for Perlin noise-based food generation."""

    scale: float = 10.0
    max_prob: float = 0.0


class Team(Enum):
    RED = "RED"
    BLUE = "BLUE"


class UnitType(Enum):
    FIGHTER = "FIGHTER"
    SCOUT = "SCOUT"


# What can be observed at a cell
@dataclass(frozen=True)
class Empty:
    pass


@dataclass(frozen=True)
class UnitPresent:
    team: Team
    unit_id: UnitId
    unit_type: UnitType = UnitType.FIGHTER


@dataclass(frozen=True)
class BasePresent:
    team: Team


@dataclass(frozen=True)
class FoodPresent:
    count: int


CellContents = Empty | UnitPresent | BasePresent | FoodPresent


RawObservations = dict[Pos, list[CellContents]]



def _generate_random_base(
    seed_pos: Pos,
    target_size: int,
    grid_width: int = GRID_SIZE,
    grid_height: int = GRID_SIZE,
) -> Region:
    """Generate a random base region by growing from a seed position.

    Args:
        seed_pos: Starting position for the base
        target_size: Number of cells in the final base (default: 12)
        grid_width: Width of the grid (default: GRID_SIZE)
        grid_height: Height of the grid (default: GRID_SIZE)

    Returns:
        A Region containing exactly target_size cells
    """
    cells = {seed_pos}

    while len(cells) < target_size:
        # Get all cells that are neighbors of current base cells
        candidates = set()
        for cell in cells:
            neighbors = [
                Pos(cell.x + 1, cell.y),
                Pos(cell.x - 1, cell.y),
                Pos(cell.x, cell.y + 1),
                Pos(cell.x, cell.y - 1),
            ]
            for neighbor in neighbors:
                # Check if neighbor is on the grid and not already in the base
                if (
                    0 <= neighbor.x < grid_width
                    and 0 <= neighbor.y < grid_height
                    and neighbor not in cells
                ):
                    candidates.add(neighbor)

        if not candidates:
            # Should never happen with reasonable target_size and grid size
            break

        # Pick a random candidate and add it to the base
        new_cell = random.choice(list(candidates))
        cells.add(new_cell)

    return Region(frozenset(cells))


def _generate_food(
    config: FoodConfig, grid_width: int = GRID_SIZE, grid_height: int = GRID_SIZE
) -> dict[Pos, int]:
    """Generate food locations using Perlin noise with Poisson-distributed counts.

    Args:
        config: FoodConfig with noise parameters
        grid_width: Width of the grid (default: GRID_SIZE)
        grid_height: Height of the grid (default: GRID_SIZE)

    Returns:
        A dict mapping Pos to food count at that position
    """

    noise = perlin(
        grid_width,
        grid_height,
    )

    # Perlin noise returns values in range [-1, 1], normalize to [0, 1]
    normalized_value = (noise + 1) / 2

    # Map to mean range [0, max_prob] for Poisson distribution
    # max_prob now represents the maximum mean for the Poisson distribution
    poisson_mean = normalized_value * config.max_prob

    # Generate food count using Poisson distribution
    food_counts_grid = np.random.poisson(poisson_mean)

    return {
        Pos(x, y): int(food_counts_grid[x, y])
        for x in range(grid_width)
        for y in range(grid_height)
        if food_counts_grid[x, y] > 0
    }


class Mind(ABC):
    @abstractmethod
    def act(self, body: Unit) -> UnitStep:
        ...
    def observe(self, body: Unit, observations: RawObservations) -> None:
        ...

@dataclass
class Unit:
    team: Team
    mind: Mind
    pos: Pos
    # original_pos: Pos  # Where the unit spawned (for returning home)
    unit_type: UnitType = UnitType.FIGHTER
    clock: Timestamp = 0
    # plan: Plan = field(default_factory=Plan)
    # # Observations since last sync with home base
    # observation_log: ObservationLog = field(default_factory=dict)
    # last_sync_tick: Timestamp = 0
    carrying_food: int = 0
    id: UnitId = field(default_factory=generate_unit_id)

    @property
    def visibility_radius(self) -> int:
        """Get visibility radius based on unit type."""
        if self.unit_type == UnitType.SCOUT:
            return 10  # Scouts have 2x visibility
        return 5  # Fighters have normal visibility

    # def home_pos(self) -> Pos:
    #     """Get this unit's original spawn position (for returning home)."""
    #     return self.original_pos

    def is_in_base(self, state: "GameState") -> bool:
        """Check if unit is inside its home base region."""
        return state.get_base_region(self.team).contains(self.pos)


def make_game(
    *,
    make_mind: Callable[[], Mind],
    grid_width: int = GRID_SIZE,
    grid_height: int = GRID_SIZE,
    init_base_size: int = 5,
    food_config: FoodConfig = FoodConfig(),
) -> GameState:
    red_base_seed = Pos(
        random.randint(0, grid_width // 2 - 1), random.randint(0, grid_height - 1)
    )
    blue_base_seed = Pos(
        random.randint(grid_width // 2, grid_width - 1),
        random.randint(0, grid_height - 1),
    )
    red_base = _generate_random_base(
        red_base_seed,
        target_size=init_base_size,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    blue_base = _generate_random_base(
        blue_base_seed,
        target_size=init_base_size,
        grid_width=grid_width,
        grid_height=grid_height,
    )

    # Create starting units: 1 fighter and 1 scout per team
    red_positions = random.sample(list(red_base.cells), 2)
    blue_positions = random.sample(list(blue_base.cells), 2)

    units_list = [
        Unit(Team.RED, make_mind(), red_positions[0], unit_type=UnitType.FIGHTER),
        # Unit(Team.RED, make_mind(), red_positions[1], unit_type=UnitType.SCOUT),
        Unit(Team.BLUE, make_mind(), blue_positions[0], unit_type=UnitType.FIGHTER),
        # Unit(Team.BLUE, make_mind(), blue_positions[1], unit_type=UnitType.SCOUT),
    ]
    return GameState(
        grid_width=grid_width,
        grid_height=grid_height,
        base_regions={Team.RED: red_base, Team.BLUE: blue_base},
        units={unit.id: unit for unit in units_list},
        food=_generate_food(
            food_config, grid_width=grid_width, grid_height=grid_height
        ),
    )


class UnitStep(Protocol):
    def execute(self, unit: Unit, state: "GameState") -> None:
        ...

@dataclass(frozen=True)
class NoopStep:
    def execute(self, unit: Unit, state: "GameState") -> None:
        pass

@dataclass(frozen=True)
class MoveStep:
    direction: Literal["up", "down", "left", "right"]

    def execute(self, unit: Unit, state: "GameState") -> None:
        dx, dy = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0),
        }[self.direction]
        unit.pos = Pos(unit.pos.x + dx, unit.pos.y + dy)

        enemies = [other for other in state.units.values() if other.pos == unit.pos and other.team != unit.team]
        enemy_fighter = [e for e in enemies if e.unit_type == UnitType.FIGHTER][:1]
        if unit.unit_type == UnitType.FIGHTER and enemies:
            kill_unit(state, enemies[0].id)
        if enemy_fighter:
            kill_unit(state, unit.id)
            return

        # If there's food at the new position, pick it up (unless it's in your own base)
        if unit.pos not in state.get_base_region(unit.team).cells:
            unit.carrying_food += state.food.pop(unit.pos, 0)


        # If carrying food and in base, drop it onto the ground
        for team in Team:
            base_region = state.get_base_region(team)
            for u in state.units.values():
                if u.team == team and base_region.contains(u.pos) and u.carrying_food > 0:
                    state.food.setdefault(u.pos, 0)
                    state.food[u.pos] += u.carrying_food
                    u.carrying_food = 0



def kill_unit(state: "GameState", unit_id: UnitId) -> None:
    # breakpoint()
    unit = state.units.pop(unit_id)
    if unit.carrying_food > 0:
        state.food.setdefault(unit.pos, 0)
        state.food[unit.pos] += unit.carrying_food


class PlayerAction(Protocol):
    def execute(self, state: "GameState", team: Team) -> None:
        ...

@dataclass(frozen=True)
class CreateUnitPlayerAction:
    mind: Mind
    unit_type: UnitType

    def execute(self, state: "GameState", team: Team) -> None:
        base = state.get_base_region(team)
        for pos in base.cells:
            if state.food.get(pos, 0) > 0:
                state.food[pos] -= 1
                if state.food[pos] <= 0:
                    del state.food[pos]
                state.add_unit(Unit(team, self.mind, pos, self.unit_type, clock=state.now))
                return

@dataclass
class GameState:
    grid_width: int
    grid_height: int
    base_regions: dict[Team, Region]

    now: Timestamp = 0
    units: dict[UnitId, Unit] = field(default_factory=dict)
    food: dict[Pos, int] = field(default_factory=dict)

    def apply_player_actions(self, player_actions: dict[Team, list[PlayerAction]]) -> None:
        for team, actions in player_actions.items():
            for action in actions:
                action.execute(self, team)

    def tick(self) -> None:
        self.now += 1
        for unit in list(self.units.values()):
            unit.clock = self.now
            step = unit.mind.act(unit)
            step.execute(unit, self)

        for unit in self.units.values():
            unit.mind.observe(unit, self.observe_from_position(unit.pos, unit.visibility_radius))

    def get_base_region(self, team: Team) -> Region:
        """Get a team's base region."""
        return self.base_regions[team]

    # def set_unit_plan(self, unit_id: UnitId, plan: Plan) -> None:
    #     unit = self.units[unit_id]
    #     if not unit.is_in_base(self):
    #         raise ValueError(f"Unit {unit_id} is not in base")
    #     unit.plan = plan

    def add_unit(self, unit: Unit) -> None:
        # breakpoint()
        self.units[unit.id] = unit

    def get_food_count_in_base(self, team: Team) -> int:
        """Get the total amount of food in a team's base."""
        base_region = self.get_base_region(team)
        return sum(
            count for pos, count in self.food.items() if base_region.contains(pos)
        )

    def observe_from_position(
        self, observer_pos: Pos, visibility_radius: int
    ) -> RawObservations:
        """Return what can be observed from a given position."""
        observations: RawObservations = {}

        # Check all positions within visibility radius
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                if abs(dx) + abs(dy) <= visibility_radius:
                    pos = Pos(observer_pos.x + dx, observer_pos.y + dy)
                    if 0 <= pos.x < self.grid_width and 0 <= pos.y < self.grid_height:
                        # Check what's at this position
                        contents = self._view_contents_at(pos)
                        observations[pos] = contents

        return observations

    def _view_contents_at(self, pos: Pos) -> list[CellContents]:
        """Determine what's actually at a position right now."""
        contents: list[CellContents] = []

        # Check for units
        for unit in self.units.values():
            if unit.pos == pos:
                contents.append(UnitPresent(unit.team, unit.id, unit.unit_type))

        # Check for bases (any cell in the region)
        if self.get_base_region(Team.RED).contains(pos):
            contents.append(BasePresent(Team.RED))
        if self.get_base_region(Team.BLUE).contains(pos):
            contents.append(BasePresent(Team.BLUE))

        # Check for food
        if pos in self.food:
            contents.append(FoodPresent(count=self.food[pos]))

        # If nothing found, return empty
        if not contents:
            contents.append(Empty())

        return contents
