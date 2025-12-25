"""Tests for knowledge and trajectory computation."""

import pytest
from typing import cast
from core import Pos, Region, Timestamp
from mechanics import (
    Team,
    Unit,
    UnitId,
    GameState,
)
from planning import (
    Move,
    Plan,
    Interrupt,
    MoveThereAction,
    EnemyInRangeCondition,
    PlanningMind,
)
from test_utils import make_unit
from knowledge import compute_expected_trajectory, ExpectedTrajectory, PlayerKnowledge


def make_simple_knowledge(
    grid_width: int = 20,
    grid_height: int = 20,
) -> PlayerKnowledge:
    """Create a simple player knowledge for testing."""
    return PlayerKnowledge(
        team=Team.RED,
        grid_width=grid_width,
        grid_height=grid_height,
        tick=Timestamp(0),
    )


class TestComputeExpectedTrajectory:
    def test_trajectory_for_unit_with_move_order_goes_to_dest(self) -> None:
        """Unit with a move order should have trajectory leading to destination."""
        target = Pos(10, 8)
        unit = make_unit(Team.RED, Pos(5, 5), plan=Plan(orders=[Move(target=target)]))
        knowledge = make_simple_knowledge()

        trajectory = compute_expected_trajectory(unit, knowledge, start_tick=0)

        # First position should be starting position
        assert trajectory[0] == Pos(5, 5)
        # Should eventually reach the target
        final_tick = max(trajectory.keys())
        assert trajectory[final_tick] == target

    def test_trajectory_runs_for_full_max_ticks_even_after_arrival(
        self,
    ) -> None:
        """Trajectory should run for full max_ticks, with unit staying at destination after arrival."""
        target = Pos(8, 5)  # Only 3 steps away
        unit = make_unit(Team.RED, Pos(5, 5), plan=Plan(orders=[Move(target=target)]))
        knowledge = make_simple_knowledge()

        trajectory = compute_expected_trajectory(
            unit, knowledge, start_tick=0, max_ticks=100
        )

        # Should have 101 entries: initial + 100 ticks
        assert len(trajectory) == 101
        # Unit should reach target at tick 3
        assert trajectory[3] == target
        # Unit should stay at target for remaining ticks
        assert all(trajectory[t] == target for t in range(3, 101))

    def test_trajectory_for_unit_with_no_orders_stays_at_current_pos(
        self,
    ) -> None:
        """Unit with no orders should stay at current position for full max_ticks."""
        unit = make_unit(Team.RED, Pos(5, 5), plan=Plan(orders=[]))  # No orders
        knowledge = make_simple_knowledge()

        trajectory = compute_expected_trajectory(unit, knowledge, start_tick=0, max_ticks=100)

        # Should have 101 entries: initial + 100 ticks
        assert len(trajectory) == 101
        # All positions should be the starting position
        assert all(pos == Pos(5, 5) for pos in trajectory.values())

    def test_trajectory_with_multiple_sequential_orders(self) -> None:
        """Unit with multiple move orders should follow them in sequence."""
        waypoint1 = Pos(3, 0)
        waypoint2 = Pos(3, 3)
        unit = make_unit(Team.RED, Pos(0, 0), plan=Plan(orders=[Move(target=waypoint1), Move(target=waypoint2)]))
        knowledge = make_simple_knowledge()

        trajectory = compute_expected_trajectory(
            unit, knowledge, start_tick=0, max_ticks=100
        )

        # Should have 101 entries: initial + 100 ticks
        assert len(trajectory) == 101
        # Should start at origin
        assert trajectory[0] == Pos(0, 0)
        # Should pass through waypoint1
        assert waypoint1 in trajectory.values()
        # Should reach waypoint2 at tick 6 (0,0 -> 3,0 is 3 steps, 3,0 -> 3,3 is 3 steps)
        assert trajectory[6] == waypoint2
        # Should stay at waypoint2 for remaining ticks
        assert all(trajectory[t] == waypoint2 for t in range(6, 101))

    def test_trajectory_respects_max_ticks_limit(self) -> None:
        """Trajectory should stop at max_ticks even if order not complete."""
        far_target = Pos(50, 50)  # Very far away
        unit = make_unit(Team.RED, Pos(0, 0), plan=Plan(orders=[Move(target=far_target)]))
        knowledge = make_simple_knowledge(grid_width=100, grid_height=100)

        max_ticks = 10
        trajectory = compute_expected_trajectory(
            unit, knowledge, start_tick=0, max_ticks=max_ticks
        )

        # Should have at most max_ticks + 1 entries (initial + max_ticks steps)
        assert len(trajectory) <= max_ticks + 1
        # Should not have reached the target
        final_tick = max(trajectory.keys())
        assert trajectory[final_tick] != far_target

    def test_trajectory_for_horizontal_movement_is_straight_line(self) -> None:
        """Unit moving purely horizontally should follow expected path."""
        target = Pos(4, 5)
        unit = make_unit(Team.RED, Pos(0, 5), plan=Plan(orders=[Move(target=target)]))
        knowledge = make_simple_knowledge()

        trajectory = compute_expected_trajectory(unit, knowledge, start_tick=0, max_ticks=100)

        # Should have 101 entries: initial + 100 ticks
        assert len(trajectory) == 101
        # First 5 ticks should be the straight line (0,5) -> (1,5) -> (2,5) -> (3,5) -> (4,5)
        assert [trajectory[t] for t in range(5)] == [Pos(i, 5) for i in range(5)]
        # Remaining positions should all be at target
        assert all(trajectory[t] == target for t in range(4, 101))
