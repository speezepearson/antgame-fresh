"""Tests for game mechanics."""

import random
from typing import Iterable, Mapping
import pytest
from core import Pos, Region
from mechanics import (
    Interrupt,
    MoveThereAction,
    Team,
    Unit,
    GameState,
    Move,
    Plan,
    EnemyInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodInRangeCondition,
    UnitPresent,
    BasePresent,
    FoodPresent,
    Empty,
    tick_game,
    CellContents,
    make_unit,
    _generate_unit_id,
)


def make_simple_game(
    grid_width: int = 10,
    grid_height: int = 10,
    red_base: Region | None = None,
    blue_base: Region | None = None,
    units: Iterable[Unit] = (),
    food: Mapping[Pos, int] = {},
) -> GameState:
    units = list(units)
    posns = [*[u.pos for u in units], *food.keys(), Pos(grid_width-1, grid_height-1)]
    return GameState(
        grid_width=max(p.x for p in posns)+1,
        grid_height=max(p.y for p in posns)+1,
        base_regions={
            Team.RED: Region(cells=frozenset(red_base.cells) if red_base is not None else frozenset({Pos(0,0)})),
            Team.BLUE: Region(cells=frozenset(blue_base.cells) if blue_base is not None else frozenset({Pos(grid_width-1, grid_height-1)})),
        },
        units=units,
        food=dict(food),
    )


class TestMove:
    def test_completes_when_unit_reaches_target(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        move = Move(target=Pos(5, 5))
        assert move.is_complete(unit)

    def test_not_complete_when_unit_away_from_target(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0))
        move = Move(target=Pos(5, 5))
        assert not move.is_complete(unit)

    def test_moves_unit_horizontally(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 5))
        move = Move(target=Pos(3, 5))
        state = make_simple_game()

        move.execute_step(unit, state)
        assert unit.pos == Pos(1, 5)

    def test_moves_unit_vertically(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 0))
        move = Move(target=Pos(5, 3))
        state = make_simple_game()

        move.execute_step(unit, state)
        assert unit.pos == Pos(5, 1)

    def test_moves_unit_closer_when_moving_diagonally(self) -> None:
        unit = make_unit(Team.RED, Pos(0, 0))
        target = Pos(5, 5)
        move = Move(target=target)
        state = make_simple_game()

        initial_distance = unit.pos.manhattan_distance(target)
        move.execute_step(unit, state)
        final_distance = unit.pos.manhattan_distance(target)

        assert final_distance < initial_distance

    def test_does_not_move_when_already_at_target(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        move = Move(target=Pos(5, 5))
        state = make_simple_game()

        move.execute_step(unit, state)
        assert unit.pos == Pos(5, 5)


class TestEnemyInRangeCondition:
    def test_fires_when_enemy_is_close(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [UnitPresent(Team.BLUE, _generate_unit_id())]  # Distance 2
        }
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_for_distant_enemy(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(10, 10): [UnitPresent(Team.BLUE, _generate_unit_id())]  # Distance 10
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_for_nearby_ally(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [UnitPresent(Team.RED, _generate_unit_id())]  # Distance 1, same team
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_when_no_units_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {Pos(6, 5): [Empty()]}
        assert not condition.evaluate(unit, observations)

    def test_fires_when_enemy_at_exact_range(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(8, 5): [UnitPresent(Team.BLUE, _generate_unit_id())]  # Distance exactly 3
        }
        assert condition.evaluate(unit, observations)


class TestBaseVisibleCondition:
    def test_fires_when_own_base_is_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(2, 16): [BasePresent(Team.RED)]
        }
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_for_enemy_base(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(29, 16): [BasePresent(Team.BLUE)]
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_when_no_base_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {Pos(10, 10): [Empty()]}
        assert not condition.evaluate(unit, observations)


class TestPositionReachedCondition:
    def test_fires_when_unit_at_exact_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = PositionReachedCondition(position=Pos(5, 5))
        observations: dict[Pos, list[CellContents]] = {}
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_when_unit_at_different_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = PositionReachedCondition(position=Pos(10, 10))
        observations: dict[Pos, list[CellContents]] = {}
        assert not condition.evaluate(unit, observations)


class TestPlan:
    def test_returns_first_order_as_current(self) -> None:
        move1 = Move(target=Pos(1, 1))
        move2 = Move(target=Pos(2, 2))
        plan = Plan(orders=[move1, move2])
        assert plan.current_order() == move1

    def test_returns_none_when_order_queue_is_empty(self) -> None:
        plan = Plan(orders=[])
        assert plan.current_order() is None

    def test_removes_current_order_when_completed(self) -> None:
        move1 = Move(target=Pos(1, 1))
        move2 = Move(target=Pos(2, 2))
        plan = Plan(orders=[move1, move2])

        plan.complete_current_order()
        assert plan.current_order() == move2
        assert len(plan.orders) == 1

    def test_replaces_orders_when_interrupted(self) -> None:
        move1 = Move(target=Pos(1, 1))
        move2 = Move(target=Pos(2, 2))
        interrupt_order = Move(target=Pos(9, 9))
        plan = Plan(orders=[move1, move2])

        plan.interrupt_with([interrupt_order])
        assert plan.orders == [interrupt_order]


    # def test_base_region_has_init_base_size_cells(self) -> None:
    #     random.seed(42)
    #     state = GameState(init_base_size=6)
    #     region = state.get_base_region(Team.RED)
    #     assert len(region.cells) == 6


class TestGameStateGetContentsAt:
    def test_returns_empty_for_empty_cell(self) -> None:
        state = make_simple_game()
        # Clear units for this test
        contents = state._get_contents_at(Pos(5, 5))
        assert len(contents) == 1
        assert isinstance(contents[0], Empty)

    def test_returns_unit_when_unit_at_position(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        state = make_simple_game(units=[unit])

        contents = state._get_contents_at(Pos(5, 5))
        assert any(isinstance(c, UnitPresent) and c.team == Team.RED for c in contents)

    def test_returns_base_when_position_in_base_region(self) -> None:
        state = make_simple_game()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))

        contents = state._get_contents_at(base_cell)
        assert any(isinstance(c, BasePresent) and c.team == Team.RED for c in contents)

    def test_returns_both_unit_and_base_when_unit_in_base(self) -> None:
        state = make_simple_game()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        unit = make_unit(Team.RED, base_cell)
        state.units = [unit]

        contents = state._get_contents_at(base_cell)
        has_unit = any(isinstance(c, UnitPresent) for c in contents)
        has_base = any(isinstance(c, BasePresent) for c in contents)
        assert has_unit and has_base


class TestGameStateObserveFromPosition:
    def test_observes_positions_within_visibility_radius(self) -> None:
        state = make_simple_game(grid_width=20, grid_height=20)
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # Should observe positions within visibility_radius (8)
        assert Pos(10, 10) in observations  # Self
        assert Pos(18, 10) in observations  # 8 away horizontally
        assert Pos(10, 18) in observations  # 8 away vertically

    def test_does_not_observe_beyond_visibility_radius(self) -> None:
        state = make_simple_game()
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # Should not observe positions beyond visibility_radius (8)
        assert Pos(19, 10) not in observations  # 9 away
        assert Pos(10, 19) not in observations  # 9 away

    def test_respects_grid_boundaries(self) -> None:
        state = make_simple_game()
        observer_pos = Pos(0, 0)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # All observed positions should be on the grid
        for pos in observations.keys():
            assert 0 <= pos.x < state.grid_width
            assert 0 <= pos.y < state.grid_height


class TestTickGame:
    def test_units_execute_movement_orders(self) -> None:
        state = make_simple_game(units=[make_unit(Team.RED, Pos(0, 0))])
        unit = state.units[0]
        initial_pos = unit.pos
        target = Pos(initial_pos.x + 3, initial_pos.y)
        unit.plan = Plan(orders=[Move(target=target)])

        tick_game(state)

        # Unit should have moved one step closer
        assert unit.pos.manhattan_distance(target) < initial_pos.manhattan_distance(
            target
        )

    def test_completes_and_removes_finished_orders(self) -> None:
        state = make_simple_game(units=[make_unit(Team.RED, Pos(0, 0))])
        unit = state.units[0]
        # Give an already-complete order
        unit.plan = Plan(orders=[Move(target=unit.pos)])

        tick_game(state)

        # Order should be removed
        assert len(unit.plan.orders) == 0

    def test_interrupts_trigger_when_condition_met(self) -> None:
        red_unit = make_unit(Team.RED, Pos(10, 10))
        blue_unit = make_unit(Team.BLUE, Pos(12, 10))
        state = make_simple_game(units=[red_unit, blue_unit])

        # Position blue unit near red unit
        red_unit.plan.interrupts = [
            Interrupt(
                condition=EnemyInRangeCondition(distance=3), actions=[MoveThereAction()]
            )
        ]

        tick_game(state)

        current = red_unit.plan.current_order()
        assert isinstance(current, Move)
        assert current.target == blue_unit.pos

    def test_syncs_logbook_when_unit_reaches_base(self) -> None:
        red_unit = make_unit(Team.RED, Pos(10, 10))
        state = make_simple_game(units=[red_unit])
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        red_unit.pos = base_cell
        initial_tick = state.tick
        red_unit.observation_log[initial_tick] = {base_cell: [Empty()]}

        tick_game(state)

        # Unit's logbook should be synced to base and cleared during tick
        assert len(red_unit.observation_log) == 0
        # Sync timestamp should be updated
        assert red_unit.last_sync_tick == initial_tick


class TestMutualAnnihilation:
    def test_opposing_units_on_same_cell_mutually_annihilate(self) -> None:
        """When a RED and BLUE unit occupy the same cell, both should be destroyed."""
        state = make_simple_game()
        red_unit = make_unit(Team.RED, Pos(5, 5))
        blue_unit = make_unit(Team.BLUE, Pos(5, 5))
        state.units = [red_unit, blue_unit]

        tick_game(state)

        # Both units should be destroyed
        assert len(state.units) == 0

    def test_opposing_units_moving_to_same_cell_mutually_annihilate(self) -> None:
        """When units from opposing teams move onto the same cell, they should destroy each other."""
        state = make_simple_game()
        red_unit = make_unit(Team.RED, Pos(5, 5))
        blue_unit = make_unit(Team.BLUE, Pos(7, 5))

        # Make blue unit move toward red unit
        blue_unit.plan.orders = [Move(target=Pos(5, 5))]
        state.units = [red_unit, blue_unit]

        # Execute one tick - blue moves one step closer
        tick_game(state)

        # Blue should have moved to (6, 5)
        assert len(state.units) == 2
        assert blue_unit.pos == Pos(6, 5)

        # Execute another tick - blue moves to same cell as red
        tick_game(state)

        # Both units should be destroyed
        assert len(state.units) == 0

    def test_allied_units_on_same_cell_do_not_annihilate(self) -> None:
        """When units from the same team occupy the same cell, they should not be destroyed."""
        state = make_simple_game()
        red_unit1 = make_unit(Team.RED, Pos(5, 5))
        red_unit2 = make_unit(Team.RED, Pos(5, 5))
        state.units = [red_unit1, red_unit2]

        tick_game(state)

        # Both units should still exist
        assert len(state.units) == 2

    def test_multiple_opposing_units_on_same_cell_all_annihilate(self) -> None:
        """When multiple units from different teams are on the same cell, all should be destroyed."""
        state = make_simple_game()
        red_unit1 = make_unit(Team.RED, Pos(5, 5))
        red_unit2 = make_unit(Team.RED, Pos(5, 5))
        blue_unit1 = make_unit(Team.BLUE, Pos(5, 5))
        blue_unit2 = make_unit(Team.BLUE, Pos(5, 5))
        state.units = [red_unit1, red_unit2, blue_unit1, blue_unit2]

        tick_game(state)

        # All units should be destroyed
        assert len(state.units) == 0

    def test_annihilation_at_one_position_does_not_affect_units_elsewhere(self) -> None:
        """Mutual annihilation at one position should not affect units at other positions."""
        state = make_simple_game()
        # Units at (5, 5) that will annihilate
        red_unit1 = make_unit(Team.RED, Pos(5, 5))
        blue_unit1 = make_unit(Team.BLUE, Pos(5, 5))
        # Units at other positions that should survive
        red_unit2 = make_unit(Team.RED, Pos(10, 10))
        blue_unit2 = make_unit(Team.BLUE, Pos(15, 15))
        state.units = [red_unit1, blue_unit1, red_unit2, blue_unit2]

        tick_game(state)

        # Only the two units not at (5, 5) should remain
        assert len(state.units) == 2
        assert red_unit2 in state.units
        assert blue_unit2 in state.units
        assert red_unit1 not in state.units
        assert blue_unit1 not in state.units


class TestFoodInRange:
    def test_fires_when_food_is_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)]  # Distance 2
        }
        assert condition.evaluate(unit, observations) == Pos(7, 5)

    def test_returns_nearest_food_when_multiple_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)],  # Distance 2
            Pos(10, 10): [FoodPresent(count=1)],  # Distance 10
            Pos(6, 6): [FoodPresent(count=1)],  # Distance 2
        }
        result = condition.evaluate(unit, observations)
        # Should return one of the closest foods (distance 2)
        assert result in [Pos(7, 5), Pos(6, 6)]

    def test_does_not_fire_when_no_food_visible(self) -> None:
        unit = make_unit(Team.RED, Pos(5, 5))
        condition = FoodInRangeCondition(distance=5)
        observations: dict[Pos, list[CellContents]] = {Pos(6, 5): [Empty()]}
        assert condition.evaluate(unit, observations) is None


class TestFoodObservation:
    def test_food_appears_in_observations(self) -> None:
        state = make_simple_game()
        state.food[Pos(10, 10)] = 3

        contents = state._get_contents_at(Pos(10, 10))
        assert any(isinstance(c, FoodPresent) and c.count == 3 for c in contents)

    def test_no_food_when_position_has_no_food(self) -> None:
        state = make_simple_game()

        contents = state._get_contents_at(Pos(10, 10))
        assert not any(isinstance(c, FoodPresent) for c in contents)


class TestUnitsCarryFood:
    def test_food_vanishes_when_unit_steps_onto_it(self) -> None:
        state = make_simple_game()
        unit = make_unit(Team.RED, Pos(0, 0))
        state.units.append(unit)
        state.food[Pos(0, 2)] = 2

        unit.plan = Plan(orders=[Move(target=Pos(0, 2))])

        tick_game(state)
        assert state.food.get(Pos(0, 2)) == 2

        tick_game(state)
        tick_game(state)
        tick_game(state)
        assert state.food.get(Pos(0, 2), 0) == 0
        assert unit.carrying_food == 2

    def test_unit_drops_food_when_annihilated(self) -> None:
        state = make_simple_game()
        red_unit = make_unit(Team.RED, Pos(0, 0), plan=Plan())
        red_unit.carrying_food = 2
        blue_unit = make_unit(Team.BLUE, Pos(0, 2), plan=Plan())
        blue_unit.carrying_food = 3
        red_unit.plan = Plan(orders=[Move(target=blue_unit.pos)])
        state.units = [red_unit, blue_unit]

        tick_game(state)
        assert state.food.get(Pos(0, 2), 0) == 0

        tick_game(state)
        assert not state.units
        assert state.food.get(Pos(0, 2)) == 5


class TestCarriedFoodSpawnsUnits:
    def test_unit_carrying_food_in_own_base_spawns_unit_in_empty_cell(self) -> None:
        state = make_simple_game(red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})))
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        red_unit = make_unit(Team.RED, base_cell)
        red_unit.carrying_food = 1
        state.units = [red_unit]

        tick_game(state)

        assert len(state.units) == 2
        assert {unit.pos for unit in state.units} <= red_base.cells

    def test_unit_loses_food_when_unit_is_spawned(self) -> None:
        state = make_simple_game(red_base=Region(cells=frozenset({Pos(0, i) for i in range(5)})))
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        red_unit = make_unit(Team.RED, base_cell)
        red_unit.carrying_food = 2
        state.units = [red_unit]
        tick_game(state)
        assert len(state.units) == 3
        assert red_unit.carrying_food == 0

    def test_unit_carrying_food_in_enemy_base_does_not_spawn_unit(self) -> None:
        state = make_simple_game()
        blue_base = state.get_base_region(Team.BLUE)
        base_cell = next(iter(blue_base.cells))
        red_unit = make_unit(Team.RED, base_cell)
        red_unit.carrying_food = 1
        state.units = [red_unit]
        tick_game(state)
        assert state.units == [red_unit]

    def test_no_unit_spawned_when_base_full(self) -> None:
        state = make_simple_game()

        red_base = state.get_base_region(Team.RED)
        base_cells = list(red_base.cells)

        # Fill all base cells with units
        for cell in base_cells:
            state.units.append(make_unit(Team.RED, cell))

        # Place food at a base cell
        state.food[base_cells[0]] = 1

        initial_unit_count = len(state.units)
        tick_game(state)

        # No new unit should be spawned since base is full
        assert len(state.units) == initial_unit_count
        # Food should still be there
        assert state.food.get(base_cells[0]) == 1

    def test_multiple_food_items_spawn_multiple_units(self) -> None:
        red_unit = make_unit(Team.RED, Pos(0, 0))
        red_unit.carrying_food = 2
        state = make_simple_game(
            red_base=Region(cells=frozenset({Pos(0, i) for i in range(3)})),
            units=[red_unit],
        )

        tick_game(state)

        assert len(state.units) == 3
        assert red_unit.carrying_food == 0
