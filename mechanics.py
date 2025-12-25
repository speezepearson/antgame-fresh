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


def _generate_unit_id() -> UnitId:
    """Generate a unique unit ID."""
    global _next_unit_id
    uid = UnitId(_next_unit_id)
    _next_unit_id += 1
    return uid


def make_unit(
    team: Team,
    pos: Pos,
    original_pos: Pos | None = None,
    plan: Plan | None = None,
) -> Unit:
    """Create a unit with an auto-generated ID. Useful for tests."""
    return Unit(
        id=_generate_unit_id(),
        team=team,
        pos=pos,
        original_pos=original_pos if original_pos is not None else pos,
        plan=plan if plan is not None else Plan(),
    )


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


@dataclass(frozen=True)
class MoveOrder:
    target: Pos


# What can be observed at a cell
@dataclass(frozen=True)
class Empty:
    pass


@dataclass(frozen=True)
class UnitPresent:
    team: Team
    unit_id: UnitId


@dataclass(frozen=True)
class BasePresent:
    team: Team


@dataclass(frozen=True)
class FoodPresent:
    count: int


CellContents = Empty | UnitPresent | BasePresent | FoodPresent


# ===== Plan-Based Order System =====


class Order(ABC):
    """Base class for all orders that units can execute."""

    @property
    @abstractmethod
    def description(self) -> str:
        """The name of the order."""
        ...

    @abstractmethod
    def is_complete(self, unit: "Unit") -> bool:
        """Check if this order has been completed."""
        pass

    @abstractmethod
    def execute_step(self, unit: "Unit", state: "GameState") -> None:
        """Execute one step of this order."""
        pass


@dataclass
class Move(Order):
    """Move to a target position."""

    target: Pos

    @property
    def description(self) -> str:
        return f"move to ({self.target.x}, {self.target.y})"

    def is_complete(self, unit: "Unit") -> bool:
        return unit.pos == self.target

    def execute_step(self, unit: "Unit", state: "GameState") -> None:
        """Move one step toward the target."""
        if self.is_complete(unit):
            return

        dx = (
            0
            if self.target.x == unit.pos.x
            else (1 if self.target.x > unit.pos.x else -1)
        )
        dy = (
            0
            if self.target.y == unit.pos.y
            else (1 if self.target.y > unit.pos.y else -1)
        )

        if dx != 0:
            unit.pos = Pos(unit.pos.x + dx, unit.pos.y)
        elif dy != 0:
            unit.pos = Pos(unit.pos.x, unit.pos.y + dy)

        # If there's food at the new position, pick it up
        unit.carrying_food += state.food.pop(unit.pos, 0)


T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class Condition(Protocol[T_co]):
    """Protocol for conditions that can trigger interrupts.

    Returns T | None when evaluated - None means condition not met,
    otherwise returns data to pass to the interrupt action.
    """

    @property
    def description(self) -> str: ...

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> T_co | None:
        """Evaluate this condition. Returns data if condition is met, None otherwise."""
        ...


@dataclass(frozen=True)
class EnemyInRangeCondition:
    """Condition: enemy unit is within a certain distance.

    Returns the position of the first enemy found within range.
    """

    distance: int

    @property
    def description(self) -> str:
        return f"enemy within {self.distance}"

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> Pos | None:
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, UnitPresent) and contents.team != unit.team:
                    if unit.pos.manhattan_distance(pos) <= self.distance:
                        return pos
        return None

@dataclass(frozen=True)
class IdleCondition:
    """Condition: unit is idle.
    """

    @property
    def description(self) -> str:
        return "idle"

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> Literal[True] | None:
        return True if len(unit.plan.orders) == 0 else None


@dataclass(frozen=True)
class BaseVisibleCondition:
    """Condition: home base is visible.

    Returns the position of the first base cell found.
    """

    @property
    def description(self) -> str:
        return "base visible"

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> Pos | None:
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, BasePresent) and contents.team == unit.team:
                    return pos
        return None


@dataclass(frozen=True)
class PositionReachedCondition:
    """Condition: unit has reached a specific position.

    Returns the position when reached.
    """

    position: Pos

    @property
    def description(self) -> str:
        return f"reached ({self.position.x}, {self.position.y})"

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> Pos | None:
        if unit.pos == self.position:
            return self.position
        return None


@dataclass(frozen=True)
class FoodInRangeCondition:
    """Condition: food is visible within range.

    Returns the position of the nearest food found, or None if no food is visible.
    """

    distance: int

    @property
    def description(self) -> str:
        return f"food within {self.distance}"

    def evaluate(
        self, unit: "Unit", observations: dict[Pos, list[CellContents]]
    ) -> Pos | None:
        food_posns = [
            pos
            for pos, contents_list in observations.items()
            if 0 < pos.manhattan_distance(unit.pos) <= self.distance
            and any(isinstance(x, FoodPresent) for x in contents_list)
        ]
        if not food_posns:
            return None
        return min(food_posns, key=lambda pos: unit.pos.manhattan_distance(pos))


@dataclass(frozen=True)
class Action(Protocol[T_contra]):
    """A named, inspectable action that generates orders based on input data.

    The generic parameter T represents the type of data this action expects.
    Actions are typically paired with Conditions that produce matching T values.
    """

    @property
    def description(self) -> str:
        """The name of the action."""
        ...

    def execute(self, unit: "Unit", data: T_contra) -> list[Order]:
        """Figure out what orders the unit should follow."""
        ...


@dataclass(frozen=True)
class MoveThereAction:
    description = "move there"

    def execute(self, unit: "Unit", data: Pos) -> list[Order]:
        return [Move(target=data)]


@dataclass(frozen=True)
class MoveHomeAction:
    description = "move home"

    def execute(self, unit: "Unit", data: object) -> list[Order]:
        return [Move(target=unit.home_pos())]

@dataclass(frozen=True)
class ResumeAction:
    description = "move to base"

    def execute(self, unit: "Unit", data: object) -> list[Order]:
        return unit.plan.orders


@dataclass(frozen=True)
class Interrupt(Generic[T_co]):
    """An interrupt handler that can preempt a plan when a condition is met.

    When the condition evaluates to a non-None value, that value is passed
    to the action to generate new orders.

    Design considerations:
    - Generic[T] provides type-safe construction: mypy ensures the condition's
      output type matches the action's input type at interrupt creation time.
    - Remains fully inspectable at runtime: condition and action fields can be
      examined separately for debugging, logging, and display purposes.
    - Heterogeneous storage: A Plan can hold interrupts with different T values
      (e.g., Interrupt[Pos], Interrupt[int], Interrupt[None]) by declaring
      the list as list[Interrupt[Any]]. The generic T is erased at runtime.
    - This design balances type safety (catching mismatched condition/action
      pairs at construction) with flexibility (storing mixed interrupt types).
    """

    condition: Condition[T_co]
    actions: list[Action[T_co]]

    def __str__(self) -> str:
        return f"when {self.condition}: [{'; '.join([action.description for action in self.actions])}]"


@dataclass
class Plan:
    """A plan consisting of a queue of orders and interrupt handlers."""

    orders: list[Order] = field(default_factory=list)
    interrupts: list[Interrupt[Any]] = field(default_factory=list)

    def current_order(self) -> Order | None:
        """Get the current order (first in queue)."""
        return self.orders[0] if self.orders else None

    def complete_current_order(self) -> None:
        """Remove the current order from the queue."""
        if self.orders:
            self.orders.pop(0)

    def interrupt_with(self, action: list[Order]) -> None:
        """Replace the order queue with interrupt actions."""
        self.orders = list(action)


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


RawObservations = dict[Pos, list[CellContents]]
ObservationLog = dict[Timestamp, RawObservations]


@dataclass
class Unit:
    id: UnitId
    team: Team
    pos: Pos
    original_pos: Pos  # Where the unit spawned (for returning home)
    unit_type: UnitType = UnitType.FIGHTER
    plan: Plan = field(default_factory=Plan)
    # Observations since last sync with home base
    observation_log: ObservationLog = field(default_factory=dict)
    last_sync_tick: Timestamp = 0
    carrying_food: int = 0

    @property
    def visibility_radius(self) -> int:
        """Get visibility radius based on unit type."""
        if self.unit_type == UnitType.SCOUT:
            return 10  # Scouts have 2x visibility
        return 5  # Fighters have normal visibility

    def home_pos(self) -> Pos:
        """Get this unit's original spawn position (for returning home)."""
        return self.original_pos

    def is_in_base(self, state: "GameState") -> bool:
        """Check if unit is inside its home base region."""
        return state.get_base_region(self.team).contains(self.pos)


def make_game(
    *,
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

    units_list = [
        *[
            Unit(_generate_unit_id(), Team.RED, pos, pos)
            for pos in random.sample(list(red_base.cells), 3)
        ],
        *[
            Unit(_generate_unit_id(), Team.BLUE, pos, pos)
            for pos in random.sample(list(blue_base.cells), 3)
        ],
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


@dataclass
class GameState:
    grid_width: int
    grid_height: int
    base_regions: dict[Team, Region]

    tick: Timestamp = 0
    units: dict[UnitId, Unit] = field(default_factory=dict)
    food: dict[Pos, int] = field(default_factory=dict)
    unit_disposition: dict[Team, UnitType] = field(default_factory=lambda: {Team.RED: UnitType.FIGHTER, Team.BLUE: UnitType.FIGHTER})

    def get_base_region(self, team: Team) -> Region:
        """Get a team's base region."""
        return self.base_regions[team]

    def upsert_units(self, *units: Unit) -> None:
        for unit in units:
            self.units[unit.id] = unit

    def kill_units(self, *unit_ids: UnitId) -> None:
        for unit_id in unit_ids:
            unit = self.units.pop(unit_id)
            if unit.carrying_food > 0:
                self.food.setdefault(unit.pos, 0)
                self.food[unit.pos] += unit.carrying_food

    def set_unit_plan(self, unit_id: UnitId, plan: Plan) -> None:
        unit = self.units[unit_id]
        if not unit.is_in_base(self):
            raise ValueError(f"Unit {unit_id} is not in base")
        unit.plan = plan

    def observe_from_position(
        self, observer_pos: Pos, visibility_radius: int
    ) -> dict[Pos, list[CellContents]]:
        """Return what can be observed from a given position."""
        observations: dict[Pos, list[CellContents]] = {}

        # Check all positions within visibility radius
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                if abs(dx) + abs(dy) <= visibility_radius:
                    pos = Pos(observer_pos.x + dx, observer_pos.y + dy)
                    if 0 <= pos.x < self.grid_width and 0 <= pos.y < self.grid_height:
                        # Check what's at this position
                        contents = self.get_contents_at(pos)
                        observations[pos] = contents

        return observations

    def get_contents_at(self, pos: Pos) -> list[CellContents]:
        """Determine what's actually at a position right now."""
        contents: list[CellContents] = []

        # Check for units
        for unit in self.units.values():
            if unit.pos == pos:
                contents.append(UnitPresent(unit.team, unit.id))

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


def tick_game(state: GameState) -> None:
    """Advance the game by one tick."""
    # 1. Record observations for all units
    for unit in state.units.values():
        observations = state.observe_from_position(unit.pos, unit.visibility_radius)
        unit.observation_log[state.tick] = observations

    # 2. Check interrupts for each unit
    for unit in state.units.values():
        # Get current observations for this unit
        observations = state.observe_from_position(unit.pos, unit.visibility_radius)

        # Check each interrupt condition
        for interrupt in unit.plan.interrupts:
            result = interrupt.condition.evaluate(unit, observations)
            if result is not None:
                # First matching interrupt triggers: call action with result and replace order queue
                new_orders = sum(
                    [action.execute(unit, result) for action in interrupt.actions], []
                )
                unit.plan.interrupt_with(new_orders)
                break  # Only first matching interrupt per tick

    # 3. Execute current order for each unit
    for unit in state.units.values():
        current_order = unit.plan.current_order()
        if current_order is None:
            continue

        # Execute one step of the order
        current_order.execute_step(unit, state)

        # If order is complete, remove it from queue
        if current_order.is_complete(unit):
            unit.plan.complete_current_order()

    # 3.5. Check for mutual annihilation (opposing units on same cell)
    units_by_position: dict[Pos, list[Unit]] = {}
    for unit in state.units.values():
        if unit.pos not in units_by_position:
            units_by_position[unit.pos] = []
        units_by_position[unit.pos].append(unit)

    # Remove units that share cells with enemies
    units_to_remove: set[UnitId] = set()
    for pos, units_at_pos in units_by_position.items():
        if len(units_at_pos) > 1:
            # Check if there are enemies at this position
            teams_at_pos = {unit.team for unit in units_at_pos}
            if len(teams_at_pos) > 1:
                # Multiple teams at same position - mutual annihilation
                # Scouts don't participate in combat - they don't kill enemies
                fighters_to_remove = [
                    unit.id for unit in units_at_pos if unit.unit_type != UnitType.SCOUT
                ]
                units_to_remove.update(fighters_to_remove)

    state.kill_units(*units_to_remove)

    # 3.75. Process food at bases to spawn new units
    for team in Team:
        base_region = state.get_base_region(team)
        unoccupied_cells = [
            cell
            for cell in base_region.cells
            if cell not in {unit.pos for unit in state.units.values()}
        ]
        new_units: list[Unit] = []
        for u in state.units.values():
            if u.team == team and base_region.contains(u.pos):
                while unoccupied_cells and u.carrying_food > 0:
                    closest_cell = min(
                        unoccupied_cells,
                        key=lambda cell: u.pos.manhattan_distance(cell),
                    )
                    new_units.append(
                        Unit(
                            id=_generate_unit_id(),
                            team=u.team,
                            pos=closest_cell,
                            original_pos=closest_cell,
                            unit_type=state.unit_disposition[team],
                        )
                    )
                    unoccupied_cells.remove(closest_cell)
                    u.carrying_food -= 1
        state.upsert_units(*new_units)

    state.tick += 1
