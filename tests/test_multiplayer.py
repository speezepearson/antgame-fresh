"""Tests for multiplayer functionality: serialization, server, and client."""

import pytest
import threading
import time
from typing import Any

from core import Pos, Region, Timestamp
from mechanics import (
    Team,
    Unit,
    UnitId,
    Plan,
    Move,
    GameState,
    make_game,
    Interrupt,
    EnemyInRangeCondition,
    MoveHomeAction,
    MoveThereAction,
    FoodInRangeCondition,
    Empty,
    UnitPresent,
    BasePresent,
    FoodPresent,
    BaseVisibleCondition,
    PositionReachedCondition,
    tick_game,
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


# ===== Serialization Tests =====


def test_serialize_deserialize_pos():
    """Test position serialization."""
    pos = Pos(5, 10)
    data = serialize_pos(pos)
    assert data == {"x": 5, "y": 10}
    assert deserialize_pos(data) == pos


def test_serialize_deserialize_region():
    """Test region serialization."""
    region = Region(frozenset([Pos(0, 0), Pos(1, 1), Pos(2, 2)]))
    data = serialize_region(region)
    deserialized = deserialize_region(data)
    assert deserialized.cells == region.cells


def test_serialize_deserialize_team():
    """Test team serialization."""
    assert serialize_team(Team.RED) == "RED"
    assert deserialize_team("RED") == Team.RED
    assert serialize_team(Team.BLUE) == "BLUE"
    assert deserialize_team("BLUE") == Team.BLUE


def test_serialize_deserialize_cell_contents():
    """Test cell contents serialization."""
    # Empty
    empty = Empty()
    assert deserialize_cell_contents(serialize_cell_contents(empty)) == empty

    # UnitPresent
    unit_present = UnitPresent(team=Team.RED, unit_id=UnitId(42))
    deserialized = deserialize_cell_contents(serialize_cell_contents(unit_present))
    assert deserialized.team == unit_present.team  # type: ignore[union-attr]
    assert deserialized.unit_id == unit_present.unit_id  # type: ignore[union-attr]

    # BasePresent
    base_present = BasePresent(team=Team.BLUE)
    assert deserialize_cell_contents(serialize_cell_contents(base_present)) == base_present

    # FoodPresent
    food_present = FoodPresent(count=5)
    assert deserialize_cell_contents(serialize_cell_contents(food_present)) == food_present


def test_serialize_deserialize_order():
    """Test order serialization."""
    move = Move(target=Pos(10, 20))
    data = serialize_order(move)
    deserialized = deserialize_order(data)
    assert isinstance(deserialized, Move)
    assert deserialized.target == move.target


def test_serialize_deserialize_conditions():
    """Test condition serialization."""
    # EnemyInRangeCondition
    cond1 = EnemyInRangeCondition(distance=5)
    data1 = serialize_condition(cond1)
    assert deserialize_condition(data1).distance == cond1.distance

    # BaseVisibleCondition
    cond2 = BaseVisibleCondition()
    data2 = serialize_condition(cond2)
    assert isinstance(deserialize_condition(data2), BaseVisibleCondition)

    # PositionReachedCondition
    cond3 = PositionReachedCondition(position=Pos(3, 4))
    data3 = serialize_condition(cond3)
    assert deserialize_condition(data3).position == cond3.position

    # FoodInRangeCondition
    cond4 = FoodInRangeCondition(distance=3)
    data4 = serialize_condition(cond4)
    assert deserialize_condition(data4).distance == cond4.distance


def test_serialize_deserialize_actions():
    """Test action serialization."""
    # MoveThereAction
    action1 = MoveThereAction()
    data1 = serialize_action(action1)
    assert isinstance(deserialize_action(data1), MoveThereAction)

    # MoveHomeAction
    action2 = MoveHomeAction()
    data2 = serialize_action(action2)
    assert isinstance(deserialize_action(data2), MoveHomeAction)


def test_serialize_deserialize_interrupt():
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


def test_serialize_deserialize_plan():
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


def test_serialize_deserialize_unit():
    """Test unit serialization."""
    unit = Unit(
        id=UnitId(123),
        team=Team.RED,
        pos=Pos(5, 6),
        original_pos=Pos(5, 6),
        carrying_food=3,
        visibility_radius=5,
        plan=Plan(orders=[Move(target=Pos(10, 10))]),
        observation_log={},
        last_sync_tick=Timestamp(0),
    )
    data = serialize_unit(unit)
    deserialized = deserialize_unit(data)
    assert deserialized.id == unit.id
    assert deserialized.team == unit.team
    assert deserialized.pos == unit.pos
    assert deserialized.original_pos == unit.original_pos
    assert deserialized.carrying_food == unit.carrying_food
    assert deserialized.visibility_radius == unit.visibility_radius
    assert len(deserialized.plan.orders) == 1


def test_serialize_deserialize_observation_log():
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


def test_serialize_deserialize_expected_trajectory():
    """Test expected trajectory serialization."""
    trajectory = ExpectedTrajectory(
        unit_id=UnitId(42),
        start_tick=Timestamp(10),
        positions=[Pos(0, 0), Pos(1, 1), Pos(2, 2)],
    )
    data = serialize_expected_trajectory(trajectory)
    deserialized = deserialize_expected_trajectory(data)
    assert deserialized.unit_id == trajectory.unit_id
    assert deserialized.start_tick == trajectory.start_tick
    assert deserialized.positions == trajectory.positions


def test_serialize_deserialize_player_knowledge():
    """Test player knowledge serialization."""
    knowledge = PlayerKnowledge(
        team=Team.RED,
        grid_width=32,
        grid_height=32,
        tick=Timestamp(5),
    )
    knowledge.all_observations = {
        Timestamp(0): {Pos(0, 0): [Empty()]},
    }
    knowledge.last_seen = {
        UnitId(1): (
            Timestamp(3),
            Unit(
                id=UnitId(1),
                team=Team.RED,
                pos=Pos(5, 5),
                original_pos=Pos(5, 5),
                plan=Plan(),
                observation_log={},
                last_sync_tick=Timestamp(0),
            ),
        )
    }
    knowledge.expected_trajectories = {
        UnitId(2): ExpectedTrajectory(
            unit_id=UnitId(2),
            start_tick=Timestamp(2),
            positions=[Pos(3, 3), Pos(4, 4)],
        )
    }
    knowledge.last_observations = {
        Pos(1, 1): (Timestamp(4), [FoodPresent(count=2)])
    }

    data = serialize_player_knowledge(knowledge)
    deserialized = deserialize_player_knowledge(data)

    assert deserialized.team == knowledge.team
    assert deserialized.grid_width == knowledge.grid_width
    assert deserialized.grid_height == knowledge.grid_height
    assert deserialized.tick == knowledge.tick
    assert Timestamp(0) in deserialized.all_observations
    assert UnitId(1) in deserialized.last_seen
    assert UnitId(2) in deserialized.expected_trajectories
    assert Pos(1, 1) in deserialized.last_observations


# ===== LocalClient Tests =====


def test_local_client_get_player_knowledge():
    """Test LocalClient returns correct player knowledge."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    assert client.get_player_knowledge(Team.RED) == knowledge[Team.RED]
    assert client.get_player_knowledge(Team.BLUE) == knowledge[Team.BLUE]


def test_local_client_set_unit_plan():
    """Test LocalClient can set unit plans."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    # Get a unit from RED team
    red_unit = next(u for u in state.units.values() if u.team == Team.RED)
    unit_id = red_unit.id

    # Set a plan
    new_plan = Plan(orders=[Move(target=Pos(10, 10))])
    client.set_unit_plan(Team.RED, unit_id, new_plan)

    # Verify plan was set
    assert len(state.units[unit_id].plan.orders) == 1
    assert state.units[unit_id].plan.orders[0].target == Pos(10, 10)  # type: ignore[attr-defined]


def test_local_client_get_base_region():
    """Test LocalClient returns base regions."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    assert client.get_base_region(Team.RED) == state.base_regions[Team.RED]
    assert client.get_base_region(Team.BLUE) == state.base_regions[Team.BLUE]


def test_local_client_get_current_tick():
    """Test LocalClient returns current tick."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    assert client.get_current_tick() == state.tick

    # Advance game and check tick updates
    tick_game(state)
    assert client.get_current_tick() == state.tick


def test_local_client_get_god_view():
    """Test LocalClient returns god view."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    assert client.get_god_view() == state


def test_local_client_get_available_teams():
    """Test LocalClient returns all teams."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }
    client = LocalClient(state=state, knowledge=knowledge)

    teams = client.get_available_teams()
    assert Team.RED in teams
    assert Team.BLUE in teams


# ===== Server Tests =====


@pytest.fixture
def test_app():
    """Create a test Flask app with game state."""
    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }

    # Initialize knowledge with base observations
    for team in [Team.RED, Team.BLUE]:
        knowledge[team].record_observations_from_bases(state)

    server = GameServer(state, knowledge, port=5000)
    # Use Flask test client instead of starting actual server
    app = server.app
    app.config['TESTING'] = True

    yield app.test_client(), state, knowledge

    # No cleanup needed - test client doesn't start threads


def test_server_get_knowledge(test_app):
    """Test server /knowledge endpoint."""
    client, state, knowledge = test_app

    # Request knowledge for RED team
    response = client.get("/knowledge/RED")
    assert response.status_code == 200

    data = response.get_json()
    assert "knowledge" in data
    assert "base_region" in data

    # Verify knowledge data
    knowledge_data = data["knowledge"]
    assert knowledge_data["team"] == "RED"
    assert knowledge_data["grid_width"] == 16
    assert knowledge_data["grid_height"] == 16


def test_server_get_knowledge_invalid_team(test_app):
    """Test server /knowledge endpoint with invalid team."""
    client, state, knowledge = test_app

    response = client.get("/knowledge/INVALID")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_server_get_knowledge_with_tick(test_app):
    """Test server /knowledge endpoint with tick parameter."""
    client, state, knowledge = test_app

    # Advance game before making request
    tick_game(state)
    for k in knowledge.values():
        k.tick_knowledge(state)

    # Request knowledge for tick 1
    response = client.get("/knowledge/RED?tick=1")
    assert response.status_code == 200

    data = response.get_json()
    assert data["knowledge"]["tick"] >= 1


def test_server_get_knowledge_returns_immediately_for_past_tick(test_app):
    """Test server returns immediately if knowledge already at requested tick."""
    client, state, knowledge = test_app

    # Advance to tick 2
    tick_game(state)
    tick_game(state)
    for k in knowledge.values():
        k.tick_knowledge(state)
        k.tick_knowledge(state)

    # Request tick 1 (should return immediately)
    start_time = time.time()
    response = client.get("/knowledge/RED?tick=1")
    elapsed = time.time() - start_time

    assert response.status_code == 200
    assert elapsed < 0.5  # Should be nearly instant
    assert response.get_json()["knowledge"]["tick"] >= 1


def test_server_set_unit_plan(test_app):
    """Test server /act endpoint."""
    client, state, knowledge = test_app

    # Get a RED unit in the base
    red_units = [u for u in state.units.values() if u.team == Team.RED]
    unit_id = red_units[0].id

    # Create a plan
    plan_data = {
        "orders": [{"type": "Move", "target": {"x": 10, "y": 10}}],
        "interrupts": [],
    }

    # Set the plan
    response = client.post(
        f"/act/RED/{unit_id}",
        json=plan_data,
    )
    assert response.status_code == 200
    assert response.get_json()["success"] is True

    # Verify plan was set
    assert len(state.units[unit_id].plan.orders) == 1
    assert state.units[unit_id].plan.orders[0].target == Pos(10, 10)


def test_server_set_unit_plan_invalid_team(test_app):
    """Test server /act endpoint with invalid team."""
    client, state, knowledge = test_app

    plan_data = {
        "orders": [{"type": "Move", "target": {"x": 10, "y": 10}}],
        "interrupts": [],
    }

    response = client.post(
        "/act/INVALID/1",
        json=plan_data,
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_server_set_unit_plan_invalid_data(test_app):
    """Test server /act endpoint with invalid plan data."""
    client, state, knowledge = test_app

    # Invalid plan data
    response = client.post(
        "/act/RED/1",
        json={"invalid": "data"},
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_server_set_unit_plan_unit_not_in_base(test_app):
    """Test server rejects plan for unit not in base."""
    client, state, knowledge = test_app

    # Get a RED unit and move it out of base
    red_unit = next(u for u in state.units.values() if u.team == Team.RED)

    # Move unit far from base
    red_unit.pos = Pos(20, 20)

    plan_data = {
        "orders": [{"type": "Move", "target": {"x": 10, "y": 10}}],
        "interrupts": [],
    }

    response = client.post(
        f"/act/RED/{red_unit.id}",
        json=plan_data,
    )
    # Should fail because unit is not in base
    assert response.status_code == 400
    assert "error" in response.get_json()


# ===== RemoteClient Tests (Integration - requires real server) =====


@pytest.fixture
def running_server():
    """Create and start a real server for integration tests.

    Only use this for tests that actually need a running server (RemoteClient tests).
    """
    import random

    # Use random port to avoid conflicts
    port = random.randint(5100, 5999)

    state = make_game(grid_width=16, grid_height=16)
    knowledge = {
        Team.RED: PlayerKnowledge(Team.RED, 16, 16, state.tick),
        Team.BLUE: PlayerKnowledge(Team.BLUE, 16, 16, state.tick),
    }

    # Initialize knowledge with base observations
    for team in [Team.RED, Team.BLUE]:
        knowledge[team].record_observations_from_bases(state)

    server = GameServer(state, knowledge, port=port)
    server.start()

    # Give server time to start
    time.sleep(0.5)

    yield server, state, knowledge, port

    # Cleanup
    server.stop()
    time.sleep(0.3)


@pytest.mark.integration
def test_remote_client_initialization(running_server):
    """Test RemoteClient can initialize and fetch initial knowledge."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    # Verify initial knowledge was fetched
    assert client._current_knowledge is not None
    assert client._current_knowledge.team == Team.RED
    assert client._current_knowledge.grid_width == 16
    assert client._current_knowledge.grid_height == 16
    assert client._base_region is not None


@pytest.mark.integration
def test_remote_client_get_player_knowledge(running_server):
    """Test RemoteClient can fetch player knowledge."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    # Get knowledge for own team
    k = client.get_player_knowledge(Team.RED)
    assert k.team == Team.RED
    assert k.grid_width == 16
    assert k.grid_height == 16

    # Try to get knowledge for other team (should fail)
    with pytest.raises(ValueError, match="Can only view own team"):
        client.get_player_knowledge(Team.BLUE)


@pytest.mark.integration
def test_remote_client_set_unit_plan(running_server):
    """Test RemoteClient can set unit plans."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    # Get a RED unit
    red_units = [u for u in state.units.values() if u.team == Team.RED]
    unit_id = red_units[0].id

    # Set a plan
    new_plan = Plan(orders=[Move(target=Pos(12, 12))])
    client.set_unit_plan(Team.RED, unit_id, new_plan)

    # Verify plan was set on server
    assert len(state.units[unit_id].plan.orders) == 1
    assert state.units[unit_id].plan.orders[0].target == Pos(12, 12)


@pytest.mark.integration
def test_remote_client_set_unit_plan_wrong_team(running_server):
    """Test RemoteClient cannot set plans for other team."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    new_plan = Plan(orders=[Move(target=Pos(12, 12))])

    with pytest.raises(ValueError, match="Can only control own team"):
        client.set_unit_plan(Team.BLUE, UnitId(1), new_plan)


@pytest.mark.integration
def test_remote_client_get_base_region(running_server):
    """Test RemoteClient can get base region."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    region = client.get_base_region(Team.RED)
    assert region is not None
    assert len(region.cells) > 0

    # Try to get other team's region (should fail)
    with pytest.raises(ValueError, match="Can only view own team"):
        client.get_base_region(Team.BLUE)


@pytest.mark.integration
def test_remote_client_get_current_tick(running_server):
    """Test RemoteClient can get current tick."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    tick = client.get_current_tick()
    assert tick >= 0


@pytest.mark.integration
def test_remote_client_get_god_view(running_server):
    """Test RemoteClient cannot get god view."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    assert client.get_god_view() is None


@pytest.mark.integration
def test_remote_client_get_available_teams(running_server):
    """Test RemoteClient only returns own team."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    teams = client.get_available_teams()
    assert teams == [Team.RED]


@pytest.mark.integration
def test_remote_client_knowledge_updates(running_server):
    """Test RemoteClient can fetch updated knowledge."""
    server, state, knowledge, port = running_server

    client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)

    initial_tick = client.get_current_tick()

    # Advance game on server
    tick_game(state)
    for k in knowledge.values():
        k.tick_knowledge(state)

    # Fetch updated knowledge
    updated_knowledge = client.get_player_knowledge(Team.RED)
    assert updated_knowledge.tick > initial_tick


# ===== Integration Tests =====


@pytest.mark.integration
def test_client_server_integration(running_server):
    """Test full client-server interaction."""
    server, state, knowledge, port = running_server

    # Create clients for both teams
    red_client = RemoteClient(url=f"http://localhost:{port}", team=Team.RED)
    blue_client = RemoteClient(url=f"http://localhost:{port}", team=Team.BLUE)

    # Get initial knowledge
    red_knowledge = red_client.get_player_knowledge(Team.RED)
    blue_knowledge = blue_client.get_player_knowledge(Team.BLUE)

    assert red_knowledge.team == Team.RED
    assert blue_knowledge.team == Team.BLUE

    # Set plans for units
    red_unit = next(u for u in state.units.values() if u.team == Team.RED)
    blue_unit = next(u for u in state.units.values() if u.team == Team.BLUE)

    red_client.set_unit_plan(Team.RED, red_unit.id, Plan(orders=[Move(target=Pos(8, 8))]))
    blue_client.set_unit_plan(
        Team.BLUE, blue_unit.id, Plan(orders=[Move(target=Pos(9, 9))])
    )

    # Verify plans were set
    assert state.units[red_unit.id].plan.orders[0].target == Pos(8, 8)
    assert state.units[blue_unit.id].plan.orders[0].target == Pos(9, 9)

    # Advance game
    tick_game(state)
    for k in knowledge.values():
        k.tick_knowledge(state)

    # Fetch updated knowledge
    updated_red = red_client.get_player_knowledge(Team.RED)
    updated_blue = blue_client.get_player_knowledge(Team.BLUE)

    assert updated_red.tick > red_knowledge.tick
    assert updated_blue.tick > blue_knowledge.tick
