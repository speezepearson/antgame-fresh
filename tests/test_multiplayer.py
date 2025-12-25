"""Tests for multiplayer functionality: serialization, server, and client."""

import pytest
import responses
import threading
import time
from typing import Any, AsyncIterator, Generator, Iterator, cast
from aiohttp.test_utils import TestClient, TestServer

from core import Pos, Region, Timestamp
from mechanics import (
    Team,
    Unit,
    UnitId,
    GameState,
    make_game,
    Empty,
    UnitPresent,
    BasePresent,
    FoodPresent,
)
from planning import (
    Plan,
    Move,
    Interrupt,
    EnemyInRangeCondition,
    MoveHomeAction,
    MoveThereAction,
    FoodInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    PlanningMind,
)
from knowledge import PlayerKnowledge, ExpectedTrajectory
from serialization import (
    serialize_pos,
    deserialize_pos,
    serialize_region,
    deserialize_region,
    serialize_team,
    deserialize_team,
    serialize_cell_contents,
    deserialize_cell_contents,
    serialize_order,
    deserialize_order,
    serialize_condition,
    deserialize_condition,
    serialize_action,
    deserialize_action,
    serialize_interrupt,
    deserialize_interrupt,
    serialize_plan,
    deserialize_plan,
    serialize_unit,
    deserialize_unit,
    serialize_observation_log,
    deserialize_observation_log,
    serialize_expected_trajectory,
    deserialize_expected_trajectory,
    serialize_player_knowledge,
    deserialize_player_knowledge,
)
from client import LocalClient, RemoteClient, GameClient
from server import GameServer
from test_utils import make_unit


# ===== Serialization Tests =====


class TestSerialization:
    """Tests for serialization/deserialization of game objects."""

    def test_pos_roundtrips(self):
        """Test position serialization."""
        pos = Pos(5, 10)
        data = serialize_pos(pos)
        assert data == {"x": 5, "y": 10}
        assert deserialize_pos(data) == pos

    def test_region_roundtrips(self):
        """Test region serialization."""
        region = Region(frozenset([Pos(0, 0), Pos(1, 1), Pos(2, 2)]))
        data = serialize_region(region)
        deserialized = deserialize_region(data)
        assert deserialized.cells == region.cells

    def test_team_roundtrips(self):
        """Test team serialization."""
        assert serialize_team(Team.RED) == "RED"
        assert deserialize_team("RED") == Team.RED
        assert serialize_team(Team.BLUE) == "BLUE"
        assert deserialize_team("BLUE") == Team.BLUE

    def test_empty_cell_contents_roundtrips(self):
        """Test Empty cell contents serialization."""
        empty = Empty()
        assert deserialize_cell_contents(serialize_cell_contents(empty)) == empty

    def test_unit_present_cell_contents_roundtrips(self):
        """Test UnitPresent cell contents serialization."""
        unit_present = UnitPresent(team=Team.RED, unit_id=UnitId(42))
        deserialized = deserialize_cell_contents(serialize_cell_contents(unit_present))
        assert deserialized.team == unit_present.team  # type: ignore[union-attr]
        assert deserialized.unit_id == unit_present.unit_id  # type: ignore[union-attr]

    def test_base_present_cell_contents_roundtrips(self):
        """Test BasePresent cell contents serialization."""
        base_present = BasePresent(team=Team.BLUE)
        assert deserialize_cell_contents(serialize_cell_contents(base_present)) == base_present

    def test_food_present_cell_contents_roundtrips(self):
        """Test FoodPresent cell contents serialization."""
        food_present = FoodPresent(count=5)
        assert deserialize_cell_contents(serialize_cell_contents(food_present)) == food_present

    def test_order_roundtrips(self):
        """Test order serialization."""
        move = Move(target=Pos(10, 20))
        data = serialize_order(move)
        deserialized = deserialize_order(data)
        assert isinstance(deserialized, Move)
        assert deserialized.target == move.target

    def test_enemy_in_range_condition_roundtrips(self):
        """Test EnemyInRangeCondition serialization."""
        cond = EnemyInRangeCondition(distance=5)
        data = serialize_condition(cond)
        assert deserialize_condition(data).distance == cond.distance

    def test_base_visible_condition_roundtrips(self):
        """Test BaseVisibleCondition serialization."""
        cond = BaseVisibleCondition()
        data = serialize_condition(cond)
        assert isinstance(deserialize_condition(data), BaseVisibleCondition)

    def test_position_reached_condition_roundtrips(self):
        """Test PositionReachedCondition serialization."""
        cond = PositionReachedCondition(position=Pos(3, 4))
        data = serialize_condition(cond)
        assert deserialize_condition(data).position == cond.position

    def test_food_in_range_condition_roundtrips(self):
        """Test FoodInRangeCondition serialization."""
        cond = FoodInRangeCondition(distance=3)
        data = serialize_condition(cond)
        assert deserialize_condition(data).distance == cond.distance

    def test_move_there_action_roundtrips(self):
        """Test MoveThereAction serialization."""
        action = MoveThereAction()
        data = serialize_action(action)
        assert isinstance(deserialize_action(data), MoveThereAction)

    def test_move_home_action_roundtrips(self):
        """Test MoveHomeAction serialization."""
        action = MoveHomeAction()
        data = serialize_action(action)
        assert isinstance(deserialize_action(data), MoveHomeAction)

    def test_interrupt_roundtrips(self):
        """Test interrupt serialization."""
        interrupt = Interrupt(
            condition=EnemyInRangeCondition(distance=2),
            actions=[MoveHomeAction(), MoveThereAction()],
        )
        data = serialize_interrupt(interrupt)
        deserialized = deserialize_interrupt(data)
        assert isinstance(deserialized.condition, EnemyInRangeCondition)
        assert deserialized.condition.distance == 2
        assert len(deserialized.actions) == 2
        assert isinstance(deserialized.actions[0], MoveHomeAction)
        assert isinstance(deserialized.actions[1], MoveThereAction)

    def test_plan_roundtrips(self):
        """Test plan serialization."""
        plan = Plan(
            orders=[Move(target=Pos(1, 2)), Move(target=Pos(3, 4))],
            interrupts=[
                Interrupt(
                    condition=FoodInRangeCondition(distance=5),
                    actions=[MoveThereAction()],
                )
            ],
        )
        data = serialize_plan(plan)
        deserialized = deserialize_plan(data)
        assert len(deserialized.orders) == 2
        assert deserialized.orders[0].target == Pos(1, 2)  # type: ignore[attr-defined]
        assert deserialized.orders[1].target == Pos(3, 4)  # type: ignore[attr-defined]
        assert len(deserialized.interrupts) == 1
        assert isinstance(deserialized.interrupts[0].condition, FoodInRangeCondition)

    def test_unit_roundtrips(self):
        """Test unit serialization."""
        unit = make_unit(Team.RED, Pos(5, 6), plan=Plan(orders=[Move(target=Pos(10, 10))]))
        unit.carrying_food = 3
        # Override the auto-generated ID for test consistency
        from mechanics import UnitId
        unit.id = UnitId(123)

        data = serialize_unit(unit)
        deserialized = deserialize_unit(data)
        assert deserialized.id == unit.id
        assert deserialized.team == unit.team
        assert deserialized.pos == unit.pos
        assert deserialized.carrying_food == unit.carrying_food
        assert deserialized.visibility_radius == unit.visibility_radius
        # Check that the mind is a PlanningMind with the plan
        mind = cast(PlanningMind, deserialized.mind)
        assert len(mind.plan.orders) == 1

    def test_observation_log_roundtrips(self):
        """Test observation log serialization."""
        log = {
            Timestamp(0): {
                Pos(0, 0): [Empty()],
                Pos(1, 1): [UnitPresent(team=Team.RED, unit_id=UnitId(1))],
            },
            Timestamp(5): {
                Pos(2, 2): [FoodPresent(count=3)],
            },
        }
        data = serialize_observation_log(log)  # type: ignore[arg-type]
        deserialized = deserialize_observation_log(data)
        assert Timestamp(0) in deserialized
        assert Timestamp(5) in deserialized
        assert Pos(0, 0) in deserialized[Timestamp(0)]
        assert Pos(2, 2) in deserialized[Timestamp(5)]

    def test_expected_trajectory_roundtrips(self):
        """Test expected trajectory serialization."""
        trajectory: ExpectedTrajectory = {
            Timestamp(0): Pos(0, 0),
            Timestamp(1): Pos(1, 1),
            Timestamp(2): Pos(2, 2),
        }
        data = serialize_expected_trajectory(trajectory)
        deserialized = deserialize_expected_trajectory(data)
        assert deserialized == trajectory

    def test_player_knowledge_roundtrips(self):
        """Test player knowledge serialization."""
        knowledge = PlayerKnowledge(
            team=Team.RED,
            grid_width=32,
            grid_height=32,
            tick=Timestamp(5),
        )
        # Add observations to the logbook
        knowledge.logbook.observation_log = {
            Timestamp(0): {Pos(0, 0): [Empty()]},
        }
        knowledge.last_in_base = {
            UnitId(1): (
                Timestamp(3),
                make_unit(Team.RED, Pos(5, 5)),
            )
        }
        knowledge.expected_trajectories = {
            UnitId(2): {
                Timestamp(2): Pos(3, 3),
                Timestamp(3): Pos(4, 4),
            }
        }
        knowledge.own_units_in_base = {
            UnitId(1): make_unit(Team.RED, Pos(5, 5)),
        }

        data = serialize_player_knowledge(knowledge)
        deserialized = deserialize_player_knowledge(data)

        assert deserialized.team == knowledge.team
        assert deserialized.grid_width == knowledge.grid_width
        assert deserialized.grid_height == knowledge.grid_height
        assert deserialized.tick == knowledge.tick
        assert Timestamp(0) in deserialized.logbook.observation_log
        assert UnitId(1) in deserialized.last_in_base
        assert UnitId(2) in deserialized.expected_trajectories


# ===== LocalClient Tests =====


class TestLocalClient:
    """Tests for LocalClient functionality."""

    def _make_client(self):
        """Create a LocalClient with test game state."""
        state = make_game(make_mind=PlanningMind, grid_width=16, grid_height=16)
        knowledge = {
            Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.now),
            Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.now),
        }
        return LocalClient(state=state, knowledge=knowledge), state, knowledge

    def test_returns_player_knowledge_for_team(self):
        """Test LocalClient returns correct player knowledge."""
        client, state, knowledge = self._make_client()

        assert client.get_player_knowledge(Team.RED, tick=state.now) == knowledge[Team.RED]
        assert client.get_player_knowledge(Team.BLUE, tick=state.now) == knowledge[Team.BLUE]

    def test_returns_base_region(self):
        """Test LocalClient returns base regions."""
        client, state, knowledge = self._make_client()

        assert client.get_base_region(Team.RED) == state.base_regions[Team.RED]
        assert client.get_base_region(Team.BLUE) == state.base_regions[Team.BLUE]

    def test_returns_current_tick(self):
        """Test LocalClient returns current tick."""
        client, state, knowledge = self._make_client()

        assert client.get_current_tick() == state.now

    def test_tick_updates_after_game_advances(self):
        """Test LocalClient tick updates when game advances."""
        client, state, knowledge = self._make_client()
        initial_tick = client.get_current_tick()

        client.tick_game()

        assert client.get_current_tick() == state.now
        assert client.get_current_tick() > initial_tick

    def test_returns_god_view(self):
        """Test LocalClient returns god view."""
        client, state, knowledge = self._make_client()

        assert client.get_god_view() == state

    def test_returns_all_teams_as_available(self):
        """Test LocalClient returns all teams."""
        client, state, knowledge = self._make_client()

        teams = client.get_available_teams()
        assert Team.RED in teams
        assert Team.BLUE in teams


# ===== Server Tests =====


class TestGameServer:
    """Tests for the GameServer HTTP endpoints."""

    @pytest.fixture
    async def test_app(self) -> AsyncIterator[tuple[Any, GameState, dict[Team, PlayerKnowledge], LocalClient]]:
        """Create a test aiohttp app with game state."""
        state = make_game(make_mind=PlanningMind, grid_width=16, grid_height=16)
        knowledge = {
            Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.now),
            Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.now),
        }
        local_client = LocalClient(state=state, knowledge=knowledge)

        # Initialize knowledge with base observations
        local_client._let_all_players_observe()

        server = GameServer(state, knowledge, port=5000)
        app = server._create_app()

        async with TestClient(TestServer(app)) as client:
            yield client, state, knowledge, local_client

    async def test_returns_knowledge_for_valid_team(self, test_app):
        """Test server /knowledge endpoint returns data for valid team."""
        client, state, knowledge, local_client = test_app

        response = await client.get("/knowledge/RED")
        assert response.status == 200

        data = await response.json()
        assert "knowledge" in data
        assert "base_region" in data

        knowledge_data = data["knowledge"]
        assert knowledge_data["team"] == "RED"
        assert knowledge_data["grid_width"] == 16
        assert knowledge_data["grid_height"] == 16

    async def test_rejects_invalid_team(self, test_app):
        """Test server /knowledge endpoint rejects invalid team."""
        client, state, knowledge, local_client = test_app

        response = await client.get("/knowledge/INVALID")
        assert response.status == 400
        assert "error" in await response.json()

    async def test_returns_knowledge_at_requested_tick(self, test_app):
        """Test server /knowledge endpoint with tick parameter."""
        client, state, knowledge, local_client = test_app

        # Advance game before making request
        local_client.tick_game()

        response = await client.get("/knowledge/RED?tick=1")
        assert response.status == 200

        data = await response.json()
        assert data["knowledge"]["tick"] >= 1

    async def test_returns_immediately_for_past_tick(self, test_app):
        """Test server returns immediately if knowledge already at requested tick."""
        client, state, knowledge, local_client = test_app

        # Advance to tick 2
        local_client.tick_game()
        local_client.tick_game()

        start_time = time.time()
        response = await client.get("/knowledge/RED?tick=1")
        elapsed = time.time() - start_time

        assert response.status == 200
        assert elapsed < 0.5  # Should be nearly instant
        assert (await response.json())["knowledge"]["tick"] >= 1

    async def test_rejects_plan_for_invalid_team(self, test_app):
        """Test server /act endpoint rejects invalid team."""
        client, state, knowledge, local_client = test_app

        plan_data = {
            "orders": [{"type": "Move", "target": {"x": 10, "y": 10}}],
            "interrupts": [],
        }

        response = await client.post(
            "/act/INVALID/1",
            json=plan_data,
        )
        assert response.status == 400
        assert "error" in await response.json()

    async def test_rejects_invalid_plan_data(self, test_app):
        """Test server /act endpoint rejects malformed plan data."""
        client, state, knowledge, local_client = test_app

        response = await client.post(
            "/act/RED/1",
            json={"invalid": "data"},
        )
        assert response.status == 400
        assert "error" in await response.json()


# ===== RemoteClient Tests (Integration - requires real server) =====


RunningServerFixture = tuple[GameServer, GameState, dict[Team, PlayerKnowledge], int, LocalClient]


@pytest.fixture
def running_server() -> Iterator[RunningServerFixture]:
    """Create and start a real server for integration tests."""
    import random

    # Use random port to avoid conflicts
    port = random.randint(5100, 5999)

    state = make_game(make_mind=PlanningMind, grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.now),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.now),
    }
    local_client = LocalClient(state=state, knowledge=knowledge)

    # Initialize knowledge with base observations
    local_client._let_all_players_observe()

    ready_event = threading.Event()
    server = GameServer(state, knowledge, port=port, ready_event=ready_event)
    server.start()
    ready_event.wait(timeout=5.0)

    yield server, state, knowledge, port, local_client

    # Cleanup
    server.stop()

@pytest.mark.integration
class TestRemoteClient:
    """Integration tests for RemoteClient (requires real server)."""

    def test_initializes_and_fetches_knowledge(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient can initialize and fetch initial knowledge."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        assert client._current_knowledge is not None
        assert client._current_knowledge.team == Team.RED
        assert client._current_knowledge.grid_width == 16
        assert client._current_knowledge.grid_height == 16
        assert client._base_region is not None

    def test_fetches_player_knowledge_for_own_team(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient can fetch player knowledge for own team."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        k = client.get_player_knowledge(Team.RED, tick=state.now)
        assert k.team == Team.RED
        assert k.grid_width == 16
        assert k.grid_height == 16

    def test_rejects_knowledge_request_for_other_team(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient rejects knowledge requests for other team."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        with pytest.raises(ValueError, match="Can only view own team"):
            client.get_player_knowledge(Team.BLUE, tick=state.now)

    def test_rejects_plan_for_other_team(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient cannot set plans for other team."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        new_plan = Plan(orders=[Move(target=Pos(12, 12))])

        with pytest.raises(ValueError, match="Can only control own team"):
            client.set_unit_plan(Team.BLUE, UnitId(1), new_plan)

    def test_gets_base_region_for_own_team(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient can get base region."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        region = client.get_base_region(Team.RED)
        assert region is not None
        assert len(region.cells) > 0

    def test_rejects_base_region_for_other_team(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient rejects base region request for other team."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        with pytest.raises(ValueError, match="Can only view own team"):
            client.get_base_region(Team.BLUE)

    def test_returns_current_tick(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient can get current tick."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        tick = client.get_current_tick()
        assert tick >= 0

    def test_god_view_not_available(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient cannot get god view."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        assert client.get_god_view() is None

    def test_only_returns_own_team_as_available(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient only returns own team."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        teams = client.get_available_teams()
        assert teams == [Team.RED]

    def test_knowledge_updates_when_game_advances(self, running_server: RunningServerFixture) -> None:
        """Test RemoteClient can fetch updated knowledge."""
        server, state, knowledge, port, local_client = running_server

        client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

        initial_tick = client.get_current_tick()

        # Advance game on server using local_client
        local_client.tick_game()

        updated_knowledge = client.get_player_knowledge(Team.RED, state.now)
        assert updated_knowledge.tick > initial_tick


# ===== Integration Tests =====


@pytest.mark.integration
class TestClientServerIntegration:
    """Full integration tests for client-server interaction."""

    def test_both_teams_can_connect_and_interact(self, running_server: RunningServerFixture) -> None:
        """Test full client-server interaction with both teams."""
        server, state, knowledge, port, local_client = running_server

        # Create clients for both teams
        red_client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)
        blue_client = RemoteClient(url=f"http://localhost:{port}", team=Team.BLUE)

        # Get initial knowledge
        red_knowledge = red_client.get_player_knowledge(Team.RED, tick=state.now)
        blue_knowledge = blue_client.get_player_knowledge(Team.BLUE, tick=state.now)

        assert red_knowledge.team == Team.RED
        assert blue_knowledge.team == Team.BLUE

    def test_clients_receive_updated_knowledge_after_game_advances(self, running_server: RunningServerFixture) -> None:
        """Test clients receive updated knowledge when game advances."""
        server, state, knowledge, port, local_client = running_server

        red_client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)
        blue_client = RemoteClient(url=f"http://localhost:{port}", team=Team.BLUE)

        red_knowledge = red_client.get_player_knowledge(Team.RED, tick=state.now)
        blue_knowledge = blue_client.get_player_knowledge(Team.BLUE, tick=state.now)

        # Advance game using local_client
        local_client.tick_game()

        # Fetch updated knowledge
        updated_red = red_client.get_player_knowledge(Team.RED, tick=state.now)
        updated_blue = blue_client.get_player_knowledge(Team.BLUE, tick=state.now)

        assert updated_red.tick > red_knowledge.tick
        assert updated_blue.tick > blue_knowledge.tick
