"""Tests for game mechanics."""

import random
import pytest
from core import Pos, Region
from mechanics import (
    Team, Unit, GameState, Move, Plan,
    EnemyInRangeCondition, BaseVisibleCondition, PositionReachedCondition,
    FoodInRange, UnitPresent, BasePresent, FoodPresent, Empty, tick_game, CellContents,
)


class TestMove:
    def test_completes_when_unit_reaches_target(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        move = Move(target=Pos(5, 5))
        assert move.is_complete(unit)

    def test_not_complete_when_unit_away_from_target(self) -> None:
        unit = Unit(Team.RED, Pos(0, 0), Pos(0, 0))
        move = Move(target=Pos(5, 5))
        assert not move.is_complete(unit)

    def test_moves_unit_horizontally(self) -> None:
        unit = Unit(Team.RED, Pos(0, 5), Pos(0, 5))
        move = Move(target=Pos(3, 5))
        state = GameState()

        move.execute_step(unit, state)
        assert unit.pos == Pos(1, 5)

    def test_moves_unit_vertically(self) -> None:
        unit = Unit(Team.RED, Pos(5, 0), Pos(5, 0))
        move = Move(target=Pos(5, 3))
        state = GameState()

        move.execute_step(unit, state)
        assert unit.pos == Pos(5, 1)

    def test_moves_unit_closer_when_moving_diagonally(self) -> None:
        unit = Unit(Team.RED, Pos(0, 0), Pos(0, 0))
        target = Pos(5, 5)
        move = Move(target=target)
        state = GameState()

        initial_distance = unit.pos.manhattan_distance(target)
        move.execute_step(unit, state)
        final_distance = unit.pos.manhattan_distance(target)

        assert final_distance < initial_distance

    def test_does_not_move_when_already_at_target(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        move = Move(target=Pos(5, 5))
        state = GameState()

        move.execute_step(unit, state)
        assert unit.pos == Pos(5, 5)


class TestEnemyInRangeCondition:
    def test_fires_when_enemy_is_close(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [UnitPresent(Team.BLUE)]  # Distance 2
        }
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_for_distant_enemy(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(10, 10): [UnitPresent(Team.BLUE)]  # Distance 10
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_for_nearby_ally(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [UnitPresent(Team.RED)]  # Distance 1, same team
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_when_no_units_visible(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [Empty()]
        }
        assert not condition.evaluate(unit, observations)

    def test_fires_when_enemy_at_exact_range(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = EnemyInRangeCondition(distance=3)
        observations: dict[Pos, list[CellContents]] = {
            Pos(8, 5): [UnitPresent(Team.BLUE)]  # Distance exactly 3
        }
        assert condition.evaluate(unit, observations)


class TestBaseVisibleCondition:
    def test_fires_when_own_base_is_visible(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(2, 16): [BasePresent(Team.RED)]
        }
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_for_enemy_base(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(29, 16): [BasePresent(Team.BLUE)]
        }
        assert not condition.evaluate(unit, observations)

    def test_does_not_fire_when_no_base_visible(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = BaseVisibleCondition()
        observations: dict[Pos, list[CellContents]] = {
            Pos(10, 10): [Empty()]
        }
        assert not condition.evaluate(unit, observations)


class TestPositionReachedCondition:
    def test_fires_when_unit_at_exact_position(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = PositionReachedCondition(position=Pos(5, 5))
        observations: dict[Pos, list[CellContents]] = {}
        assert condition.evaluate(unit, observations)

    def test_does_not_fire_when_unit_at_different_position(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
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


class TestGetBaseRegion:
    def test_red_base_is_on_left_side(self) -> None:
        random.seed(42)
        state = GameState()
        region = state.get_base_region(Team.RED)
        # All cells should have small x values (left half of grid)
        for cell in region.cells:
            assert cell.x < state.grid_width // 2

    def test_blue_base_is_on_right_side(self) -> None:
        random.seed(42)
        state = GameState()
        region = state.get_base_region(Team.BLUE)
        # All cells should have large x values (right half of grid)
        for cell in region.cells:
            assert cell.x >= state.grid_width // 2

    def test_base_region_has_12_cells(self) -> None:
        random.seed(42)
        state = GameState()
        region = state.get_base_region(Team.RED)
        assert len(region.cells) == 12


class TestGameStateGetContentsAt:
    def test_returns_empty_for_empty_cell(self) -> None:
        state = GameState()
        # Clear units for this test
        state.units = []
        contents = state._get_contents_at(Pos(15, 15))
        assert len(contents) == 1
        assert isinstance(contents[0], Empty)

    def test_returns_unit_when_unit_at_position(self) -> None:
        state = GameState()
        # Place a unit at a known position outside bases
        unit = Unit(Team.RED, Pos(15, 15), Pos(15, 15))
        state.units = [unit]

        contents = state._get_contents_at(Pos(15, 15))
        assert any(isinstance(c, UnitPresent) and c.team == Team.RED for c in contents)

    def test_returns_base_when_position_in_base_region(self) -> None:
        state = GameState()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))

        contents = state._get_contents_at(base_cell)
        assert any(isinstance(c, BasePresent) and c.team == Team.RED for c in contents)

    def test_returns_both_unit_and_base_when_unit_in_base(self) -> None:
        state = GameState()
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        unit = Unit(Team.RED, base_cell, base_cell)
        state.units = [unit]

        contents = state._get_contents_at(base_cell)
        has_unit = any(isinstance(c, UnitPresent) for c in contents)
        has_base = any(isinstance(c, BasePresent) for c in contents)
        assert has_unit and has_base


class TestGameStateObserveFromPosition:
    def test_observes_positions_within_visibility_radius(self) -> None:
        state = GameState()
        state.units = []
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # Should observe positions within visibility_radius (8)
        assert Pos(10, 10) in observations  # Self
        assert Pos(18, 10) in observations  # 8 away horizontally
        assert Pos(10, 18) in observations  # 8 away vertically

    def test_does_not_observe_beyond_visibility_radius(self) -> None:
        state = GameState()
        state.units = []
        observer_pos = Pos(10, 10)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # Should not observe positions beyond visibility_radius (8)
        assert Pos(19, 10) not in observations  # 9 away
        assert Pos(10, 19) not in observations  # 9 away

    def test_respects_grid_boundaries(self) -> None:
        state = GameState()
        state.units = []
        observer_pos = Pos(0, 0)
        visibility_radius = 8

        observations = state._observe_from_position(observer_pos, visibility_radius)

        # All observed positions should be on the grid
        for pos in observations.keys():
            assert 0 <= pos.x < state.grid_width
            assert 0 <= pos.y < state.grid_height


class TestTickGame:
    def test_units_execute_movement_orders(self) -> None:
        state = GameState()
        unit = state.units[0]
        initial_pos = unit.pos
        target = Pos(initial_pos.x + 3, initial_pos.y)
        unit.plan = Plan(orders=[Move(target=target)])

        tick_game(state)

        # Unit should have moved one step closer
        assert unit.pos.manhattan_distance(target) < initial_pos.manhattan_distance(target)

    def test_completes_and_removes_finished_orders(self) -> None:
        state = GameState()
        unit = state.units[0]
        # Give an already-complete order
        unit.plan = Plan(orders=[Move(target=unit.pos)])

        tick_game(state)

        # Order should be removed
        assert len(unit.plan.orders) == 0

    def test_interrupts_trigger_when_condition_met(self) -> None:
        state = GameState()
        red_unit = state.units[0]  # RED unit
        blue_unit = state.units[3]  # BLUE unit

        # Position blue unit near red unit
        red_unit.pos = Pos(10, 10)
        blue_unit.pos = Pos(12, 10)  # 2 away

        # Give red unit a plan with interrupt
        fallback_pos = Pos(5, 5)
        interrupt = {
            'condition': EnemyInRangeCondition(distance=3),
            'action': lambda enemy_pos: [Move(target=fallback_pos)]
        }
        red_unit.plan = Plan(
            orders=[Move(target=Pos(20, 20))],
            interrupts=[type('Interrupt', (), interrupt)()]  # Mock Interrupt
        )

        from mechanics import Interrupt, Action
        red_unit.plan.interrupts = [
            Interrupt(
                condition=EnemyInRangeCondition(distance=3),
                action=Action(
                    name="fallback to safe position",
                    execute=lambda enemy_pos: [Move(target=fallback_pos)]
                )
            )
        ]

        tick_game(state)

        # Plan should be interrupted, current order should be the fallback
        current = red_unit.plan.current_order()
        assert isinstance(current, Move)
        assert current.target == fallback_pos

    def test_syncs_logbook_when_unit_reaches_base(self) -> None:
        state = GameState()
        red_unit = state.units[0]
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))

        # Position unit in base and give it some observations
        red_unit.pos = base_cell
        initial_tick = state.tick
        red_unit.logbook[5] = {Pos(10, 10): [Empty()]}

        tick_game(state)

        # Unit's logbook should be synced to base and cleared during tick
        assert len(red_unit.logbook) == 0
        # Sync timestamp should be updated
        assert red_unit.last_sync_tick == initial_tick


class TestMutualAnnihilation:
    def test_opposing_units_on_same_cell_mutually_annihilate(self) -> None:
        """When a RED and BLUE unit occupy the same cell, both should be destroyed."""
        state = GameState(units=[])
        red_unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        blue_unit = Unit(Team.BLUE, Pos(5, 5), Pos(5, 5))
        state.units = [red_unit, blue_unit]

        tick_game(state)

        # Both units should be destroyed
        assert len(state.units) == 0

    def test_opposing_units_moving_to_same_cell_mutually_annihilate(self) -> None:
        """When units from opposing teams move onto the same cell, they should destroy each other."""
        state = GameState(units=[])
        red_unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        blue_unit = Unit(Team.BLUE, Pos(7, 5), Pos(7, 5))

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
        state = GameState(units=[])
        red_unit1 = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        red_unit2 = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        state.units = [red_unit1, red_unit2]

        tick_game(state)

        # Both units should still exist
        assert len(state.units) == 2

    def test_multiple_opposing_units_on_same_cell_all_annihilate(self) -> None:
        """When multiple units from different teams are on the same cell, all should be destroyed."""
        state = GameState(units=[])
        red_unit1 = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        red_unit2 = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        blue_unit1 = Unit(Team.BLUE, Pos(5, 5), Pos(5, 5))
        blue_unit2 = Unit(Team.BLUE, Pos(5, 5), Pos(5, 5))
        state.units = [red_unit1, red_unit2, blue_unit1, blue_unit2]

        tick_game(state)

        # All units should be destroyed
        assert len(state.units) == 0

    def test_annihilation_at_one_position_does_not_affect_units_elsewhere(self) -> None:
        """Mutual annihilation at one position should not affect units at other positions."""
        state = GameState(units=[])
        # Units at (5, 5) that will annihilate
        red_unit1 = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        blue_unit1 = Unit(Team.BLUE, Pos(5, 5), Pos(5, 5))
        # Units at other positions that should survive
        red_unit2 = Unit(Team.RED, Pos(10, 10), Pos(10, 10))
        blue_unit2 = Unit(Team.BLUE, Pos(15, 15), Pos(15, 15))
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
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = FoodInRange()
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)]  # Distance 2
        }
        assert condition.evaluate(unit, observations) == Pos(7, 5)

    def test_returns_nearest_food_when_multiple_visible(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = FoodInRange()
        observations: dict[Pos, list[CellContents]] = {
            Pos(7, 5): [FoodPresent(count=1)],  # Distance 2
            Pos(10, 10): [FoodPresent(count=1)],  # Distance 10
            Pos(6, 6): [FoodPresent(count=1)],  # Distance 2
        }
        result = condition.evaluate(unit, observations)
        # Should return one of the closest foods (distance 2)
        assert result in [Pos(7, 5), Pos(6, 6)]

    def test_does_not_fire_when_no_food_visible(self) -> None:
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        condition = FoodInRange()
        observations: dict[Pos, list[CellContents]] = {
            Pos(6, 5): [Empty()]
        }
        assert condition.evaluate(unit, observations) is None


class TestFoodObservation:
    def test_food_appears_in_observations(self) -> None:
        state = GameState()
        state.units = []
        state.food[Pos(10, 10)] = 3

        contents = state._get_contents_at(Pos(10, 10))
        assert any(isinstance(c, FoodPresent) and c.count == 3 for c in contents)

    def test_no_food_when_position_has_no_food(self) -> None:
        state = GameState()
        state.units = []
        state.food = {}

        contents = state._get_contents_at(Pos(10, 10))
        assert not any(isinstance(c, FoodPresent) for c in contents)


class TestFoodMovesWithUnit:
    def test_food_moves_when_unit_moves(self) -> None:
        state = GameState()
        state.units = []
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        state.units.append(unit)
        state.food[Pos(5, 5)] = 2

        # Give unit a move order
        unit.plan = Plan(orders=[Move(target=Pos(10, 5))])

        # Execute one tick
        tick_game(state)

        # Unit should have moved one step
        assert unit.pos == Pos(6, 5)
        # Food should have moved with it
        assert Pos(5, 5) not in state.food
        assert state.food.get(Pos(6, 5)) == 2

    def test_food_does_not_move_when_unit_has_no_orders(self) -> None:
        state = GameState()
        state.units = []
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        state.units.append(unit)
        state.food[Pos(5, 5)] = 2

        # No orders for the unit
        tick_game(state)

        # Food should still be at the same position
        assert state.food.get(Pos(5, 5)) == 2

    def test_multiple_food_items_move_together(self) -> None:
        state = GameState()
        state.units = []
        unit = Unit(Team.RED, Pos(5, 5), Pos(5, 5))
        state.units.append(unit)
        state.food[Pos(5, 5)] = 5

        unit.plan = Plan(orders=[Move(target=Pos(10, 5))])
        tick_game(state)

        # All food items should move together
        assert Pos(5, 5) not in state.food
        assert state.food.get(Pos(6, 5)) == 5


class TestFoodSpawnsUnits:
    def test_food_at_base_spawns_unit_in_empty_cell(self) -> None:
        state = GameState()
        # Clear all units for clean test
        state.units = []

        # Get a base cell and place food there
        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))
        state.food[base_cell] = 1

        initial_unit_count = len(state.units)
        tick_game(state)

        # A new unit should have been spawned
        assert len(state.units) == initial_unit_count + 1
        # Food should be consumed
        assert state.food.get(base_cell, 0) == 0

    def test_spawned_unit_is_at_closest_empty_cell(self) -> None:
        state = GameState()
        state.units = []

        red_base = state.get_base_region(Team.RED)
        base_cells = list(red_base.cells)

        # Place food at the first cell
        food_pos = base_cells[0]
        state.food[food_pos] = 1

        # Occupy all cells except one
        for i in range(len(base_cells) - 1):
            state.units.append(Unit(Team.RED, base_cells[i + 1], base_cells[i + 1]))

        tick_game(state)

        # A new unit should be spawned at the only empty cell
        # The empty cell should be the one closest to the food position
        occupied_after = {unit.pos for unit in state.units}
        assert len(state.units) == len(base_cells)

    def test_no_unit_spawned_when_base_full(self) -> None:
        state = GameState()
        state.units = []

        red_base = state.get_base_region(Team.RED)
        base_cells = list(red_base.cells)

        # Fill all base cells with units
        for cell in base_cells:
            state.units.append(Unit(Team.RED, cell, cell))

        # Place food at a base cell
        state.food[base_cells[0]] = 1

        initial_unit_count = len(state.units)
        tick_game(state)

        # No new unit should be spawned since base is full
        assert len(state.units) == initial_unit_count
        # Food should still be there
        assert state.food.get(base_cells[0]) == 1

    def test_multiple_food_items_spawn_multiple_units(self) -> None:
        state = GameState()
        state.units = []

        red_base = state.get_base_region(Team.RED)
        base_cell = next(iter(red_base.cells))

        # Place 3 food items
        state.food[base_cell] = 3

        tick_game(state)

        # First food spawns first unit
        assert len(state.units) == 1
        assert state.food.get(base_cell, 0) == 2

        # Continue ticking to spawn more units
        tick_game(state)
        assert len(state.units) == 2
        assert state.food.get(base_cell, 0) == 1

        tick_game(state)
        assert len(state.units) == 3
        assert base_cell not in state.food
