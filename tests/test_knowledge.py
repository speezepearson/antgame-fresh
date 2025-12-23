"""Tests for knowledge and trajectory computation."""

import pytest
from core import Pos, Region
from mechanics import (
    Team,
    Unit,
    UnitId,
    GameState,
    Move,
    Plan,
    make_unit,
    Interrupt,
    MoveThereAction,
    EnemyInRangeCondition,
)
from knowledge import compute_expected_trajectory, ExpectedTrajectory


def make_simple_game(
    grid_width: int = 20,
    grid_height: int = 20,
    units: dict[UnitId, Unit] | None = None,
) -> GameState:
    """Create a simple game state for testing."""
    return GameState(
        grid_width=grid_width,
        grid_height=grid_height,
        base_regions={
            Team.RED: Region(frozenset({Pos(0, 0)})),
            Team.BLUE: Region(frozenset({Pos(grid_width - 1, grid_height - 1)})),
        },
        units=units if units is not None else {},
        food={},
    )


class TestComputeExpectedTrajectory:
    def test_trajectory_for_unit_with_move_order_goes_to_dest(self) -> None:
        """Unit with a move order should have trajectory leading to destination."""
        unit = make_unit(Team.RED, Pos(5, 5))
        target = Pos(10, 8)
        unit.plan = Plan(orders=[Move(target=target)])
        state = make_simple_game()

        trajectory = compute_expected_trajectory(unit, state, start_tick=0)

        # First position should be starting position
        assert trajectory.positions[0] == Pos(5, 5)
        # Last position should be the target
        assert trajectory.positions[-1] == target
        # Each step should move closer or stay same distance
        for i in range(len(trajectory.positions) - 1):
            curr_dist = trajectory.positions[i].manhattan_distance(target)
            next_dist = trajectory.positions[i + 1].manhattan_distance(target)
            assert next_dist <= curr_dist

    def test_trajectory_for_unit_with_move_order_terminates_early_when_unit_arrives(
        self,
    ) -> None:
        """Trajectory should stop when unit reaches destination, not continue to max_ticks."""
        unit = make_unit(Team.RED, Pos(5, 5))
        target = Pos(8, 5)  # Only 3 steps away
        unit.plan = Plan(orders=[Move(target=target)])
        state = make_simple_game()

        trajectory = compute_expected_trajectory(
            unit, state, start_tick=0, max_ticks=100
        )

        # Should only have 4 positions: start + 3 steps
        # (5,5) -> (6,5) -> (7,5) -> (8,5)
        assert len(trajectory.positions) == 4
        assert trajectory.positions[-1] == target

    def test_trajectory_for_unit_with_no_orders_is_singleton_at_current_pos(
        self,
    ) -> None:
        """Unit with no orders should have trajectory with just its current position."""
        unit = make_unit(Team.RED, Pos(5, 5))
        unit.plan = Plan(orders=[])  # No orders
        state = make_simple_game()

        trajectory = compute_expected_trajectory(unit, state, start_tick=0)

        # Should only contain the starting position
        assert len(trajectory.positions) == 1
        assert trajectory.positions[0] == Pos(5, 5)

    def test_trajectory_with_multiple_sequential_orders(self) -> None:
        """Unit with multiple move orders should follow them in sequence."""
        unit = make_unit(Team.RED, Pos(0, 0))
        waypoint1 = Pos(3, 0)
        waypoint2 = Pos(3, 3)
        unit.plan = Plan(orders=[Move(target=waypoint1), Move(target=waypoint2)])
        state = make_simple_game()

        trajectory = compute_expected_trajectory(
            unit, state, start_tick=0, max_ticks=100
        )

        # Should start at origin
        assert trajectory.positions[0] == Pos(0, 0)
        # Should pass through or end at waypoint1
        assert waypoint1 in trajectory.positions
        # Should end at waypoint2
        assert trajectory.positions[-1] == waypoint2
        # Total distance should be manhattan distance to waypoint1 + waypoint2
        # 0,0 -> 3,0 is 3 steps, 3,0 -> 3,3 is 3 steps = 6 steps + initial pos = 7 positions
        assert len(trajectory.positions) == 7

    def test_trajectory_respects_max_ticks_limit(self) -> None:
        """Trajectory should stop at max_ticks even if order not complete."""
        unit = make_unit(Team.RED, Pos(0, 0))
        far_target = Pos(50, 50)  # Very far away
        unit.plan = Plan(orders=[Move(target=far_target)])
        state = make_simple_game(grid_width=100, grid_height=100)

        max_ticks = 10
        trajectory = compute_expected_trajectory(
            unit, state, start_tick=0, max_ticks=max_ticks
        )

        # Should have at most max_ticks + 1 positions (initial + max_ticks steps)
        assert len(trajectory.positions) <= max_ticks + 1
        # Should not have reached the target
        assert trajectory.positions[-1] != far_target

    def test_trajectory_stores_unit_id_and_start_tick(self) -> None:
        """Trajectory should correctly store metadata."""
        unit = make_unit(Team.RED, Pos(5, 5))
        unit.plan = Plan(orders=[Move(target=Pos(10, 10))])
        state = make_simple_game()
        state.tick = 42

        trajectory = compute_expected_trajectory(unit, state, start_tick=42)

        assert trajectory.unit_id == unit.id
        assert trajectory.start_tick == state.tick

    def test_trajectory_for_horizontal_movement_is_straight_line(self) -> None:
        """Unit moving purely horizontally should follow expected path."""
        unit = make_unit(Team.RED, Pos(0, 5))
        target = Pos(4, 5)
        unit.plan = Plan(orders=[Move(target=target)])
        state = make_simple_game()

        trajectory = compute_expected_trajectory(unit, state, start_tick=0)

        assert trajectory.positions == [Pos(i, 5) for i in range(5)]
