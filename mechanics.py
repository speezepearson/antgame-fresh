"""Game mechanics and state."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeVar, Generic, Callable, Any
import random
import noise  # type: ignore

from core import Timestamp, Pos, Region


# Game Constants
GRID_SIZE = 32


@dataclass
class FoodConfig:
    """Configuration for Perlin noise-based food generation."""
    scale: float = 10.0
    octaves: int = 3
    persistence: float = 0.5
    lacunarity: float = 2.0
    max_prob: float = 0.3
    seed: int | None = None


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


# ===== Plan-Based Order System =====


class Order(ABC):
    """Base class for all orders that units can execute."""

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

    def is_complete(self, unit: "Unit") -> bool:
        return unit.pos == self.target

    def execute_step(self, unit: "Unit", state: "GameState") -> None:
        """Move one step toward the target."""
        if self.is_complete(unit):
            return

        dx = 0 if self.target.x == unit.pos.x else (1 if self.target.x > unit.pos.x else -1)
        dy = 0 if self.target.y == unit.pos.y else (1 if self.target.y > unit.pos.y else -1)

        if dx != 0:
            unit.pos = Pos(unit.pos.x + dx, unit.pos.y)
        elif dy != 0:
            unit.pos = Pos(unit.pos.x, unit.pos.y + dy)


# TODO: I'm rusty on contra/covariance, but I worry that this is wrong.
T = TypeVar('T', covariant=True)


class Condition(Protocol[T]):
    """Protocol for conditions that can trigger interrupts.

    Returns T | None when evaluated - None means condition not met,
    otherwise returns data to pass to the interrupt action.
    """

    def evaluate(self, unit: "Unit", observations: dict[Pos, list[CellContents]]) -> T | None:
        """Evaluate this condition. Returns data if condition is met, None otherwise."""
        ...


@dataclass(frozen=True)
class EnemyInRangeCondition:
    """Condition: enemy unit is within a certain distance.

    Returns the position of the first enemy found within range.
    """

    distance: int

    def evaluate(self, unit: "Unit", observations: dict[Pos, list[CellContents]]) -> Pos | None:
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, UnitPresent) and contents.team != unit.team:
                    if unit.pos.manhattan_distance(pos) <= self.distance:
                        return pos
        return None


@dataclass(frozen=True)
class BaseVisibleCondition:
    """Condition: home base is visible.

    Returns the position of the first base cell found.
    """

    def evaluate(self, unit: "Unit", observations: dict[Pos, list[CellContents]]) -> Pos | None:
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

    def evaluate(self, unit: "Unit", observations: dict[Pos, list[CellContents]]) -> Pos | None:
        if unit.pos == self.position:
            return self.position
        return None


@dataclass
class Action(Generic[T]):
    """A named, inspectable action that generates orders based on input data.

    The generic parameter T represents the type of data this action expects.
    Actions are typically paired with Conditions that produce matching T values.
    """

    name: str
    execute: Callable[[T], list[Order]]

    def __str__(self) -> str:
        return self.name


@dataclass
class Interrupt(Generic[T]):
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

    condition: Condition[T]
    action: Action[T]

    def __str__(self) -> str:
        return f"when {self.condition}, do {self.action}"


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


def _generate_random_base(seed_pos: Pos, target_size: int = 12, grid_width: int = GRID_SIZE, grid_height: int = GRID_SIZE) -> Region:
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
                if (0 <= neighbor.x < grid_width and
                    0 <= neighbor.y < grid_height and
                    neighbor not in cells):
                    candidates.add(neighbor)

        if not candidates:
            # Should never happen with reasonable target_size and grid size
            break

        # Pick a random candidate and add it to the base
        new_cell = random.choice(list(candidates))
        cells.add(new_cell)

    return Region(frozenset(cells))


def _generate_food(config: FoodConfig, grid_width: int = GRID_SIZE, grid_height: int = GRID_SIZE) -> set[Pos]:
    """Generate food locations using Perlin noise.

    Args:
        config: FoodConfig with noise parameters
        grid_width: Width of the grid (default: GRID_SIZE)
        grid_height: Height of the grid (default: GRID_SIZE)

    Returns:
        A set of Pos where food is located
    """
    # Set random seed if provided
    seed = config.seed if config.seed is not None else random.randint(0, 1000000)

    food_locations = set()

    for x in range(grid_width):
        for y in range(grid_height):
            # Generate Perlin noise value for this position
            # Normalize coordinates by scale
            noise_value = noise.pnoise2(
                x / config.scale,
                y / config.scale,
                octaves=config.octaves,
                persistence=config.persistence,
                lacunarity=config.lacunarity,
                repeatx=grid_width * 10,  # Large repeat to avoid tiling
                repeaty=grid_height * 10,
                base=seed,
            )

            # Perlin noise returns values in range [-1, 1], normalize to [0, 1]
            normalized_value = (noise_value + 1) / 2

            # Map to probability range [0, max_prob]
            probability = normalized_value * config.max_prob

            # Randomly place food based on probability
            if random.random() < probability:
                food_locations.add(Pos(x, y))

    return food_locations


@dataclass
class Unit:
    team: Team
    pos: Pos
    original_pos: Pos  # Where the unit spawned (for returning home)
    plan: Plan = field(default_factory=Plan)
    # Observations since last sync with home base
    logbook: Logbook = field(default_factory=dict)
    last_sync_tick: Timestamp = 0
    visibility_radius: int = 5

    def home_pos(self) -> Pos:
        """Get this unit's original spawn position (for returning home)."""
        return self.original_pos

    def is_near_base(self, state: "GameState") -> bool:
        """Check if unit is inside its home base region."""
        return state.get_base_region(self.team).contains(self.pos)


@dataclass
class GameState:
    tick: Timestamp = 0
    units: list[Unit] = field(default_factory=list)
    selected_unit: Unit | None = None
    # Plan being constructed for the selected unit
    working_plan: Plan | None = None
    # Base regions for each team (generated randomly at game start)
    base_regions: dict[Team, Region] = field(default_factory=dict)
    # Each team's home base logbook
    base_logbooks: dict[Team, Logbook] = field(default_factory=dict)
    # Slider positions for each team's view (which tick they're viewing)
    view_tick: dict[Team, Timestamp] = field(default_factory=dict)
    # Whether each player's view auto-advances to current tick
    view_live: dict[Team, bool] = field(default_factory=dict)
    # Grid dimensions
    grid_width: int = GRID_SIZE
    grid_height: int = GRID_SIZE
    # Food locations on the grid
    food: set[Pos] = field(default_factory=set)
    # Configuration for food generation
    food_config: FoodConfig = field(default_factory=FoodConfig)

    def __post_init__(self) -> None:
        # Generate random base regions
        # RED starts on left half, BLUE on right half to keep them separated
        red_seed = Pos(
            random.randint(0, self.grid_width // 2 - 1),
            random.randint(0, self.grid_height - 1)
        )
        blue_seed = Pos(
            random.randint(self.grid_width // 2, self.grid_width - 1),
            random.randint(0, self.grid_height - 1)
        )

        self.base_regions[Team.RED] = _generate_random_base(red_seed, target_size=12, grid_width=self.grid_width, grid_height=self.grid_height)
        self.base_regions[Team.BLUE] = _generate_random_base(blue_seed, target_size=12, grid_width=self.grid_width, grid_height=self.grid_height)

        # Spawn 3 units at random positions within each base
        red_cells = list(self.base_regions[Team.RED].cells)
        blue_cells = list(self.base_regions[Team.BLUE].cells)

        red_spawns = random.sample(red_cells, 3)
        blue_spawns = random.sample(blue_cells, 3)

        for spawn_pos in red_spawns:
            self.units.append(Unit(Team.RED, spawn_pos, spawn_pos))
        for spawn_pos in blue_spawns:
            self.units.append(Unit(Team.BLUE, spawn_pos, spawn_pos))

        self.base_logbooks[Team.RED] = {}
        self.base_logbooks[Team.BLUE] = {}
        self.view_tick[Team.RED] = 0
        self.view_tick[Team.BLUE] = 0
        self.view_live[Team.RED] = True
        self.view_live[Team.BLUE] = True

        # Generate food using Perlin noise
        self.food = _generate_food(self.food_config, self.grid_width, self.grid_height)

        # Record initial observations from base regions
        self._record_observations_from_bases()

    def get_base_region(self, team: Team) -> Region:
        """Get a team's base region."""
        return self.base_regions[team]

    def get_base_pos(self, team: Team) -> Pos:
        """Get a position in the team's base (deprecated, for backward compatibility)."""
        # Return an arbitrary cell from the region
        return next(iter(self.get_base_region(team).cells))

    def _record_observations_from_bases(self) -> None:
        """Record observations from each base region and units in the base."""
        for team in [Team.RED, Team.BLUE]:
            # Start with base region observations
            observations = self._observe_from_base_region(team)

            # Add observations from units currently in the base
            for unit in self.units:
                if unit.team == team and unit.is_near_base(self):
                    unit_observations = self._observe_from_position(unit.pos, unit.visibility_radius)
                    observations.update(unit_observations)

            if observations:
                if self.tick not in self.base_logbooks[team]:
                    self.base_logbooks[team][self.tick] = {}
                self.base_logbooks[team][self.tick].update(observations)

    def _observe_from_position(self, observer_pos: Pos, visibility_radius: int) -> dict[Pos, list[CellContents]]:
        """Return what can be observed from a given position."""
        observations: dict[Pos, list[CellContents]] = {}

        # Check all positions within visibility radius
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                if abs(dx) + abs(dy) <= visibility_radius:
                    pos = Pos(observer_pos.x + dx, observer_pos.y + dy)
                    if 0 <= pos.x < self.grid_width and 0 <= pos.y < self.grid_height:
                        # Check what's at this position
                        contents = self._get_contents_at(pos)
                        observations[pos] = contents

        return observations

    def _observe_from_base_region(self, team: Team) -> dict[Pos, list[CellContents]]:
        """Observe only the tiles within the base region itself."""
        region = self.get_base_region(team)
        observations: dict[Pos, list[CellContents]] = {}
        for pos in region.cells:
            contents = self._get_contents_at(pos)
            observations[pos] = contents
        return observations

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
    # 1. Record observations for all units
    for unit in state.units:
        observations = state._observe_from_position(unit.pos, unit.visibility_radius)
        unit.logbook[state.tick] = observations

    # 2. Check interrupts for each unit
    for unit in state.units:
        # Get current observations for this unit
        observations = state._observe_from_position(unit.pos, unit.visibility_radius)

        # Check each interrupt condition
        for interrupt in unit.plan.interrupts:
            result = interrupt.condition.evaluate(unit, observations)
            if result is not None:
                # First matching interrupt triggers: call action with result and replace order queue
                new_orders = interrupt.action.execute(result)
                unit.plan.interrupt_with(new_orders)
                break  # Only first matching interrupt per tick

    # 3. Execute current order for each unit
    for unit in state.units:
        current_order = unit.plan.current_order()
        if current_order is None:
            continue

        # Execute one step of the order
        current_order.execute_step(unit, state)

        # If order is complete, remove it from queue
        if current_order.is_complete(unit):
            unit.plan.complete_current_order()

    # 4. Sync units that are at base
    for unit in state.units:
        if unit.is_near_base(state):
            state._merge_logbook_to_base(unit)

    state.tick += 1

    # Record observations from base regions for the new tick
    state._record_observations_from_bases()
