
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from core import Pos, Timestamp
from mechanics import BasePresent, CellContents, Empty, FoodPresent, GameState, Mind, MoveStep, NoopStep, RawObservations, Team, Unit, UnitId, UnitPresent, UnitStep, UnitType
from logbook import Logbook



# ===== Plan-Based Order System =====


class Order(ABC):
    """Base class for all orders that units can execute."""

    @property
    @abstractmethod
    def description(self) -> str:
        """The name of the order."""
        ...

    @abstractmethod
    def get_next_step(self, mind: "PlanningMind", body: Unit) -> UnitStep | None:
        """Execute one step of this order."""
        pass


@dataclass
class Move(Order):
    """Move to a target position."""

    target: Pos

    @property
    def description(self) -> str:
        return f"move to ({self.target.x}, {self.target.y})"

    def get_next_step(self, mind: "PlanningMind", body: Unit) -> UnitStep | None:
        """Move one step toward the target."""
        if self.target == body.pos:
            return None

        dx = self.target.x - body.pos.x
        dy = self.target.y - body.pos.y

        return MoveStep(direction=(
            ('up' if dy > 0 else 'down') if abs(dy) > abs(dx) else
            ('left' if dx < 0 else 'right')
        ))


T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

@dataclass
class PlanningMind(Mind):
    logbook: Logbook = field(default_factory=Logbook)
    plan: "Plan" = field(default_factory=lambda: Plan())

    original_pos: Pos | None = None

    def observe(self, body: Unit, observations: RawObservations) -> None:
        self.logbook.add_latest_observations(body.clock, observations)
        if self.original_pos is None:
            self.original_pos = body.pos

    def act(self, body: Unit) -> UnitStep:
        return self.plan.get_next_step(self, body)


@dataclass(frozen=True)
class SetUnitPlanPlayerAction:
    unit_id: UnitId
    plan: "Plan"

    def execute(self, state: GameState, team: Team) -> None:
        unit = state.units[self.unit_id]
        if unit.team != team:
            raise ValueError(f"Unit {self.unit_id} is not owned by team {team}")
        if not unit.is_in_base(state):
            raise ValueError(f"can't issue order to unit {self.unit_id} because it is not in base")
        cast(PlanningMind, unit.mind).plan = self.plan


class Condition(Protocol[T_co]):
    """Protocol for conditions that can trigger interrupts.

    Returns T | None when evaluated - None means condition not met,
    otherwise returns data to pass to the interrupt action.
    """

    @property
    def description(self) -> str: ...

    def evaluate(self, mind: PlanningMind, body: Unit) -> T_co | None:
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

    def evaluate(self, mind: PlanningMind, body: Unit) -> Pos | None:
        for pos, (timestamp, contents_list) in mind.logbook.last_observations_by_pos.items():
            for contents in contents_list:
                if isinstance(contents, UnitPresent) and contents.team != body.team:
                    if body.pos.manhattan_distance(pos) <= self.distance:
                        return pos
        return None

@dataclass(frozen=True)
class IdleCondition:
    """Condition: unit is idle.
    """

    @property
    def description(self) -> str:
        return "idle"

    def evaluate(self, mind: PlanningMind, body: Unit) -> Literal[True] | None:
        return True if len(mind.plan.orders) == 0 else None


@dataclass(frozen=True)
class BaseVisibleCondition:
    """Condition: home base is visible.

    Returns the position of the first base cell found.
    """

    @property
    def description(self) -> str:
        return "base visible"

    def evaluate(self, mind: PlanningMind, body: Unit) -> Pos | None:
        for pos, (timestamp, contents_list) in mind.logbook.last_observations_by_pos.items():
            for contents in contents_list:
                if isinstance(contents, BasePresent) and contents.team == body.team:
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

    def evaluate(self, unit: Unit) -> Pos | None:
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

    def evaluate(self, mind: PlanningMind, body: Unit) -> Pos | None:
        food_posns = [
            pos
            for pos, contents_list in mind.logbook.latest_observations.items()
            if 0 < pos.manhattan_distance(body.pos) <= self.distance
            and not any(isinstance(x, BasePresent) and x.team == body.team for x in contents_list)  # don't pick up food in your own base
            and any(isinstance(x, FoodPresent) for x in contents_list)
        ]
        if not food_posns:
            return None
        return min(food_posns, key=lambda pos: body.pos.manhattan_distance(pos))


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

    def execute(self, mind: PlanningMind, body: Unit, data: T_contra) -> list[Order]:
        """Figure out what orders the unit should follow."""
        ...


@dataclass(frozen=True)
class MoveThereAction:
    description = "move there"

    def execute(self, mind: PlanningMind, body: Unit, data: Pos) -> list[Order]:
        return [Move(target=data)]


@dataclass(frozen=True)
class MoveHomeAction:
    description = "move home"

    def execute(self, mind: PlanningMind, body: Unit, data: object) -> list[Order]:
        return [Move(target=mind.original_pos)] if mind.original_pos else []

@dataclass(frozen=True)
class ResumeAction:
    description = "move to base"

    def execute(self, mind: PlanningMind, body: Unit, data: object) -> list[Order]:
        return mind.plan.orders


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

    def get_next_step(self, mind: PlanningMind, body: Unit) -> UnitStep:
        for _ in range(3):
            self._check_interrupts(mind, body)
            if not self.orders:
                return NoopStep()
            step = self.orders[0].get_next_step(mind, body)
            if step is not None:
                return step
            self.orders.pop(0)
        return NoopStep()

    def _check_interrupts(self, mind: PlanningMind, body: Unit) -> None:
        for interrupt in self.interrupts:
            result = interrupt.condition.evaluate(mind, body)
            if result is not None:
                # First matching interrupt triggers: call action with result and replace order queue
                new_orders = sum(
                    [action.execute(mind, body, result) for action in interrupt.actions],
                    []
                )
                self.orders = new_orders

