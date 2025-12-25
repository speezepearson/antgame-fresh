"""Tests for game mechanics."""

import random
from typing import Iterable, Mapping, cast
import pytest
from core import Pos, Region
from mechanics import (
    Team,
    Unit,
    UnitType,
    GameState,
    UnitPresent,
    BasePresent,
    FoodPresent,
    Empty,
    CellContents,
    generate_unit_id,
)
from planning import (
    Interrupt,
    MoveThereAction,
    Move,
    Plan,
    EnemyInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodInRangeCondition,
    PlanningMind,
)
from test_utils import make_unit


def make_simple_game(
    grid_width: int = 10,
    grid_height: int = 10,
    red_base: Region | None = None,
    blue_base: Region | None = None,
    units: Iterable[Unit] = (),
    food: Mapping[Pos, int] = {},
) -> GameState:
    units_map = {u.id: u for u in units}
    posns = [
        *[u.pos for u in units_map.values()],
        *food.keys(),
        Pos(grid_width - 1, grid_height - 1),
    ]
    return GameState(
        grid_width=max(p.x for p in posns) + 1,
        grid_height=max(p.y for p in posns) + 1,
        base_regions={
            Team.RED: Region(
                cells=frozenset(red_base.cells)
                if red_base is not None
                else frozenset({Pos(0, 0)})
            ),
            Team.BLUE: Region(
                cells=frozenset(blue_base.cells)
                if blue_base is not None
                else frozenset({Pos(grid_width - 1, grid_height - 1)})
            ),
        },
        units=units_map,
        food=dict(food),
    )


class TestMove:
    def test_returns_none_when_unit_reaches_target(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        move = Move(target=Pos(5, 5))
        mind = cast(PlanningMind, unit.mind)
        assert move.get_next_step(mind, unit) is None

    def test_returns_step_when_unit_away_from_target(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0))
        move = Move(target=Pos(5, 5))
        mind = cast(PlanningMind, unit.mind)
        assert move.get_next_step(mind, unit) is not None

    def test_moves_unit_horizontally(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 5))
        state = make_simple_game(units=[unit])
        move = Move(target=Pos(3, 5))
        mind = cast(PlanningMind, unit.mind)

        step = move.get_next_step(mind, unit)
        assert step is not None
        step.execute(unit, state)
        assert unit.pos == Pos(1, 5)

    def test_moves_unit_vertically(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 0))
        state = make_simple_game(units=[unit])
        move = Move(target=Pos(5, 3))
        mind = cast(PlanningMind, unit.mind)

        step = move.get_next_step(mind, unit)
        assert step is not None
        step.execute(unit, state)
        assert unit.pos == Pos(5, 1)

    def test_moves_unit_closer_when_moving_diagonally(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0))
        target = Pos(5, 5)
        move = Move(target=target)
        state = make_simple_game(units=[unit])
        mind = cast(PlanningMind, unit.mind)

        initial_distance = unit.pos.manhattan_distance(target)
        step = move.get_next_step(mind, unit)
        assert step is not None
        step.execute(unit, state)
        final_distance = unit.pos.manhattan_distance(target)

        assert final_distance < initial_distance

    def test_does_not_move_when_already_at_target(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        move = Move(target=Pos(5, 5))
        state = make_simple_game(units=[unit])
        mind = cast(PlanningMind, unit.mind)

        step = move.get_next_step(mind, unit)
        assert step is None  # No step needed when already at target
        assert unit.pos == Pos(5, 5)

    def test_picks_up_food_when_moving_outside_own_base(self) -> None:
        # Place food outside any base
        food_pos = Pos(5, 5)
        unit = make_unit(Team.RED, Pos(4, 5), plan=Plan(orders=[Move(target=Pos(6, 5))]))
        state = make_simple_game(units=[unit], food={food_pos: 3})

        state.tick()

        assert unit.pos == food_pos
        assert unit.carrying_food == 3
        assert food_pos not in state.food

    def test_does_not_pick_up_food_in_own_base(self) -> None:
        # Place food inside red base
        food_pos = Pos(0, 2)
        unit = make_unit(Team.RED, Pos(0, 1), plan=Plan(orders=[Move(target=Pos(0, 3))]))
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})),
            units=[unit],
            food={food_pos: 3},
        )

        state.tick()

        assert unit.pos == food_pos
        assert unit.carrying_food == 0  # Did not pick up food
        assert state.food.get(food_pos) == 3  # Food still there

    def test_picks_up_food_in_enemy_base(self) -> None:
        # Place food inside blue base
        food_pos = Pos(9, 2)
        unit = make_unit(Team.RED, Pos(9, 1), plan=Plan(orders=[Move(target=Pos(9, 3))]))
        state = make_simple_game(
            blue_base=Region(cells=frozenset({Pos(9, i) for i in range(5)})),
            units=[unit],
            food={food_pos: 2},
        )

        state.tick()

        assert unit.pos == food_pos
        assert unit.carrying_food == 2  # Picked up food in enemy base
        assert food_pos not in state.food


def _set_observations_on_unit(unit: Unit, observations: dict[Pos, list[CellContents]]) -> None:
    """Helper to set observations on a unit's mind's logbook."""
    mind = cast(PlanningMind, unit.mind)
    mind.logbook.add_latest_observations(0, observations)


class TestEnemyInRangeCondition:
    def test_fires_when_enemy_is_close(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [UnitPresent(Team.BLUE, generate_unit_id())]  # Distance 2
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit)

    def test_does_not_fire_for_distant_enemy(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(10, 10): [UnitPresent(Team.BLUE, generate_unit_id())]  # Distance 10
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_does_not_fire_for_nearby_ally(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [
                UnitPresent(Team.RED, generate_unit_id())
            ]  # Distance 1, same team
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_does_not_fire_when_no_units_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {Pos(6, 5): [Empty()]}
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_fires_when_enemy_at_exact_range(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(8, 5): [
                UnitPresent(Team.BLUE, generate_unit_id())
            ]  # Distance exactly 3
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit)


class TestBaseVisibleCondition:
    def test_fires_when_own_base_is_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(2, 16): [BasePresent(Team.RED)]
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit)

    def test_does_not_fire_for_enemy_base(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(29, 16): [BasePresent(Team.BLUE)]
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_does_not_fire_when_no_base_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {Pos(10, 10): [Empty()]}
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)


class TestFoodInRangeCondition:
    def test_fires_when_food_is_nearby(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)]  # Distance 2
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit)

    def test_does_not_fire_for_distant_food(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(10, 10): [FoodPresent(count=1)]  # Distance 10
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_does_not_fire_for_food_at_unit_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(5, 5): [FoodPresent(count=1)]  # Distance 0
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_does_not_fire_for_food_in_own_base(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=3)
        # Food in red base nearby
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [FoodPresent(count=1), BasePresent(Team.RED)]  # Distance 1, in own base
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert not condition.evaluate(mind, unit)

    def test_fires_for_food_in_enemy_base(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=3)
        # Food in blue (enemy) base nearby
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [FoodPresent(count=1), BasePresent(Team.BLUE)]  # Distance 1, in enemy base
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit)


class TestPositionReachedCondition:
    def test_fires_when_unit_at_exact_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = PositionReachedCondition(position=Pos(5, 5))
        assert condition.evaluate(unit)

    def test_does_not_fire_when_unit_at_different_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = PositionReachedCondition(position=Pos(10, 10))
        assert not condition.evaluate(unit)


class TestPlan:
    def test_executes_first_order(self) -> None:
        move1 = Move(target=Pos(1, 1))
        move2 = Move(target=Pos(2, 2))
        plan = Plan(orders=[move1, move2])
        assert plan.orders[0] == move1

    def test_returns_noop_when_order_queue_is_empty(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0))
        plan = Plan(orders=[])
        mind = cast(PlanningMind, unit.mind)
        mind.plan = plan
        step = plan.get_next_step(mind, unit)
        # Should return NoopStep when no orders
        from mechanics import NoopStep
        assert isinstance(step, NoopStep)

    def test_removes_completed_order_and_moves_to_next(self) -> None:
        # Unit at (1, 1) with order to move there (already complete)
        unit = make_unit(Team.RED, Pos(1, 1))
        move1 = Move(target=Pos(1, 1))  # Already at target
        move2 = Move(target=Pos(2, 2))
        plan = Plan(orders=[move1, move2])
        mind = cast(PlanningMind, unit.mind)
        mind.plan = plan

        # Get next step should skip the completed order and return step for second order
        step = plan.get_next_step(mind, unit)
        # move1 should be removed since it's complete
        assert len(plan.orders) == 1
        assert plan.orders[0] == move2


class TestGameStateViewContentsAt:
    def test_returns_empty_for_empty_cell(self) -> None:
        state = make_simple_game()
        # Clear units for this test
        contents = state._view_contents_at(Pos(5, 5))
        assert len(contents) == 1
        assert isinstance(contents[0], Empty)

    def test_returns_unit_when_unit_at_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        state = make_simple_game(units=[unit])

        contents = state._view_contents_at(Pos(5, 5))
        assert any(isinstance(c, UnitPresent) and c.team == Team.RED for c in contents)

    def test_returns_base_when_position_in_base_region(self) -> None:
        state = make_simple_game()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))

        contents = state._view_contents_at(base_cell)
        assert any(isinstance(c, BasePresent) and c.team == Team.RED for c in contents)

    def test_returns_both_unit_and_base_when_unit_in_base(self) -> None:
        state = make_simple_game()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        unit = make_unit(Team.RED, base_cell)
        state.add_unit(unit)

        contents = state._view_contents_at(base_cell)
        has_unit = any(isinstance(c, UnitPresent) for c in contents)
        has_base = any(isinstance(c, BasePresent) for c in contents)
        assert has_unit and has_base


class TestGameStateObserveFromPosition:
    def test_observes_positions_within_visibility_radius(self) -> None:
        state = make_simple_game(grid_width=20, grid_height=20)
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state.observe_from_position(observer_pos, visibility_radius)

        # Should observe positions within visibility_radius (8)
        assert Pos(10, 10) in observations  # Self
        assert Pos(18, 10) in observations  # 8 away horizontally
        assert Pos(10, 18) in observations  # 8 away vertically

    def test_does_not_observe_beyond_visibility_radius(self) -> None:
        state = make_simple_game()
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state.observe_from_position(observer_pos, visibility_radius)

        # Should not observe positions beyond visibility_radius (8)
        assert Pos(19, 10) not in observations  # 9 away
        assert Pos(10, 19) not in observations  # 9 away

    def test_respects_grid_boundaries(self) -> None:
        state = make_simple_game()
        observer_pos = Pos(0, 0)
        visibility_radius = 8

        observations = state.observe_from_position(observer_pos, visibility_radius)

        # All observed positions should be on the grid
        for pos in observations.keys():
            assert 0 <= pos.x < state.grid_width
            assert 0 <= pos.y < state.grid_height


class TestGameTick:
    def test_units_execute_movement_orders(self) -> None:
        initial_pos = Pos(0, 0)
        target = Pos(initial_pos.x + 3, initial_pos.y)
        unit = make_unit(Team.RED, initial_pos, plan=Plan(orders=[Move(target=target)]))
        state = make_simple_game(units=[unit])

        state.tick()

        # Unit should have moved one step closer
        assert unit.pos.manhattan_distance(target) < initial_pos.manhattan_distance(
            target
        )

    def test_completes_and_removes_finished_orders(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0), plan=Plan(orders=[Move(target=Pos(0, 0))]))
        state = make_simple_game(units=[unit])

        state.tick()

        # Order should be removed
        mind = cast(PlanningMind, unit.mind)
        assert len(mind.plan.orders) == 0

    def test_interrupts_trigger_when_condition_met(self) -> None:
        red_plan = Plan(
            orders=[],
            interrupts=[
                Interrupt(
                    condition=EnemyInRangeCondition(distance=3), actions=[MoveThereAction()]
                )
            ]
        )
        red_unit = make_unit(Team.RED, Pos(10, 10), plan=red_plan)
        blue_unit = make_unit(Team.BLUE, Pos(12, 10))
        state = make_simple_game(units=[red_unit, blue_unit])

        # First tick: unit observes the enemy (observations happen AFTER acting)
        state.tick()
        # Second tick: unit acts on observations, interrupt should fire
        state.tick()

        red_mind = cast(PlanningMind, red_unit.mind)
        assert len(red_mind.plan.orders) > 0
        current = red_mind.plan.orders[0]
        assert isinstance(current, Move)
        assert current.target == Pos(12, 10)  # Original enemy position


class TestMutualAnnihilation:
    def test_opposing_units_moving_to_same_cell_mutually_annihilate(self) -> None:
        """When units from opposing teams move onto the same cell, they should destroy each other."""
        red_unit = make_unit(Team.RED, Pos(5, 5))
        blue_unit = make_unit(Team.BLUE, Pos(7, 5), plan=Plan(orders=[Move(target=Pos(5, 5))]))
        state = make_simple_game(units=[red_unit, blue_unit])

        # Execute one tick - blue moves one step closer
        state.tick()

        # Blue should have moved to (6, 5)
        assert len(state.units) == 2
        assert blue_unit.pos == Pos(6, 5)

        # Execute another tick - blue moves to same cell as red
        state.tick()

        # Both units should be destroyed
        assert len(state.units) == 0

    def test_allied_units_moving_to_same_cell_do_not_annihilate(self) -> None:
        """When units from the same team move onto the same cell, they should not be destroyed."""
        red_unit1 = make_unit(Team.RED, Pos(5, 5))
        red_unit2 = make_unit(Team.RED, Pos(6, 5), plan=Plan(orders=[Move(target=Pos(5, 5))]))
        state = make_simple_game(units=[red_unit1, red_unit2])

        state.tick()

        # Both units should still exist
        assert len(state.units) == 2
        # Both should be at same position now
        assert red_unit2.pos == Pos(5, 5)


class TestFoodInRangeReturnsNearest:
    def test_fires_when_food_is_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)]  # Distance 2
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit) == Pos(7, 5)

    def test_returns_nearest_food_when_multiple_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)],  # Distance 2
            Pos(10, 10): [FoodPresent(count=1)],  # Distance 10
            Pos(6, 6): [FoodPresent(count=1)],  # Distance 2
        }
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        result = condition.evaluate(mind, unit)
        # Should return one of the closest foods (distance 2)
        assert result in [Pos(7, 5), Pos(6, 6)]

    def test_does_not_fire_when_no_food_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {Pos(6, 5): [Empty()]}
        _set_observations_on_unit(unit, observations)
        mind = cast(PlanningMind, unit.mind)
        assert condition.evaluate(mind, unit) is None


class TestFoodObservation:
    def test_food_appears_in_observations(self) -> None:
        state = make_simple_game(food={Pos(10, 10): 3})

        contents = state._view_contents_at(Pos(10, 10))
        assert any(isinstance(c, FoodPresent) and c.count == 3 for c in contents)

    def test_no_food_when_position_has_no_food(self) -> None:
        state = make_simple_game()

        contents = state._view_contents_at(Pos(10, 10))
        assert not any(isinstance(c, FoodPresent) for c in contents)


class TestUnitsCarryFood:
    def test_food_vanishes_when_unit_steps_onto_it(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0), plan=Plan(orders=[Move(target=Pos(0, 2))]))
        state = make_simple_game(units=[unit], food={Pos(0, 2): 2})

        state.tick()
        assert state.food.get(Pos(0, 2)) == 2

        state.tick()
        assert state.food.get(Pos(0, 2), 0) == 0
        assert unit.carrying_food == 2

    def test_unit_drops_food_when_annihilated(self) -> None:
        red_unit = make_unit(Team.RED, Pos(0, 0), plan=Plan(orders=[Move(target=Pos(0, 2))]))
        red_unit.carrying_food = 2
        blue_unit = make_unit(Team.BLUE, Pos(0, 2))
        blue_unit.carrying_food = 3
        state = make_simple_game(units=[red_unit, blue_unit])

        state.tick()
        assert state.food.get(Pos(0, 2), 0) == 0

        state.tick()
        assert len(state.units) == 0
        assert state.food.get(Pos(0, 2)) == 5


class TestCarriedFoodDropsAtBase:
    def test_unit_carrying_food_moving_into_own_base_drops_food(self) -> None:
        """When a unit carrying food moves into its own base, food is dropped."""
        base_cell = Pos(0, 0)
        # Unit starts adjacent to base, will move into it
        red_unit = make_unit(Team.RED, Pos(1, 0), plan=Plan(orders=[Move(target=base_cell)]))
        red_unit.carrying_food = 1
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})),
            units=[red_unit],
        )

        state.tick()

        # Unit should have moved into base and dropped food
        assert red_unit.pos == base_cell
        assert len(state.units) == 1
        assert red_unit.carrying_food == 0
        assert state.food.get(base_cell) == 1

    def test_unit_loses_food_when_dropped_at_base(self) -> None:
        """Unit carrying multiple food items drops all when moving into base."""
        base_cell = Pos(0, 0)
        red_unit = make_unit(Team.RED, Pos(1, 0), plan=Plan(orders=[Move(target=base_cell)]))
        red_unit.carrying_food = 2
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})),
            units=[red_unit],
        )

        state.tick()

        # Food is dropped, unit count unchanged
        assert red_unit.pos == base_cell
        assert len(state.units) == 1
        assert red_unit.carrying_food == 0
        assert state.food.get(base_cell) == 2

    def test_unit_carrying_food_moving_into_enemy_base_does_not_drop_food(self) -> None:
        """Food is only dropped in own base, not enemy base."""
        enemy_base_cell = Pos(9, 9)
        red_unit = make_unit(Team.RED, Pos(8, 9), plan=Plan(orders=[Move(target=enemy_base_cell)]))
        red_unit.carrying_food = 1
        state = make_simple_game(
            blue_base=Region(cells=frozenset({enemy_base_cell})),
            units=[red_unit],
        )

        state.tick()

        # Food should not be dropped in enemy base
        assert red_unit.pos == enemy_base_cell
        assert set(state.units.keys()) == {red_unit.id}
        assert red_unit.carrying_food == 1
        assert state.food.get(enemy_base_cell) is None

    def test_multiple_food_items_dropped_at_base(self) -> None:
        """Multiple food items are all dropped when entering base."""
        base_cell = Pos(0, 0)
        red_unit = make_unit(Team.RED, Pos(1, 0), plan=Plan(orders=[Move(target=base_cell)]))
        red_unit.carrying_food = 2
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(3)})),
            units=[red_unit],
        )

        state.tick()

        # Food is dropped, unit count unchanged
        assert red_unit.pos == base_cell
        assert len(state.units) == 1
        assert red_unit.carrying_food == 0
        assert state.food.get(base_cell) == 2


class TestCreateUnitPlayerAction:
    def test_creates_unit_and_consumes_food_from_base(self) -> None:
        from mechanics import CreateUnitPlayerAction
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})),
            food={Pos(0, 0): 3},
        )

        initial_food = state.get_food_count_in_base(Team.RED)
        initial_unit_count = len(state.units)
        action = CreateUnitPlayerAction(mind=PlanningMind(), unit_type=UnitType.FIGHTER)
        action.execute(state, Team.RED)

        assert len(state.units) == initial_unit_count + 1
        new_unit = next(u for u in state.units.values() if u.team == Team.RED)
        assert new_unit.unit_type == UnitType.FIGHTER
        assert state.get_food_count_in_base(Team.RED) == initial_food - 1

    def test_does_not_create_unit_when_no_food(self) -> None:
        from mechanics import CreateUnitPlayerAction
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)}))
        )
        # No food in base

        initial_unit_count = len(state.units)
        action = CreateUnitPlayerAction(mind=PlanningMind(), unit_type=UnitType.FIGHTER)
        action.execute(state, Team.RED)

        # No new unit should be created
        assert len(state.units) == initial_unit_count

    def test_creates_correct_unit_type(self) -> None:
        from mechanics import CreateUnitPlayerAction
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})),
            food={Pos(0, 0): 2},
        )

        fighter_action = CreateUnitPlayerAction(mind=PlanningMind(), unit_type=UnitType.FIGHTER)
        fighter_action.execute(state, Team.RED)

        scout_action = CreateUnitPlayerAction(mind=PlanningMind(), unit_type=UnitType.SCOUT)
        scout_action.execute(state, Team.RED)

        units = list(state.units.values())
        assert any(u.unit_type == UnitType.FIGHTER for u in units)
        assert any(u.unit_type == UnitType.SCOUT for u in units)
