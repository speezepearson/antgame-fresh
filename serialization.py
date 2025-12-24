"""Serialization and deserialization for network communication."""

from __future__ import annotations
from typing import Any, Dict, List
import json

from core import Pos, Region, Timestamp
from mechanics import (
    Team,
    UnitId,
    Unit,
    Plan,
    Move,
    Order,
    Interrupt,
    EnemyInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodInRangeCondition,
    MoveThereAction,
    MoveHomeAction,
    Empty,
    UnitPresent,
    BasePresent,
    FoodPresent,
    CellContents,
    ObservationLog,
)
from knowledge import PlayerKnowledge, ExpectedTrajectory


# ===== Basic Types =====


def serialize_pos(pos: Pos) -> Dict[str, int]:
    return {"x": pos.x, "y": pos.y}


def deserialize_pos(data: Dict[str, Any]) -> Pos:
    return Pos(x=data["x"], y=data["y"])


def serialize_region(region: Region) -> Dict[str, List[Dict[str, int]]]:
    return {"cells": [serialize_pos(p) for p in region.cells]}


def deserialize_region(data: Dict[str, Any]) -> Region:
    cells = frozenset(deserialize_pos(p) for p in data["cells"])
    return Region(cells=cells)


def serialize_team(team: Team) -> str:
    return team.name


def deserialize_team(data: str) -> Team:
    return Team[data]


# ===== Cell Contents =====


def serialize_cell_contents(contents: CellContents) -> Dict[str, Any]:
    if isinstance(contents, Empty):
        return {"type": "Empty"}
    elif isinstance(contents, UnitPresent):
        return {
            "type": "UnitPresent",
            "team": serialize_team(contents.team),
            "unit_id": int(contents.unit_id),
        }
    elif isinstance(contents, BasePresent):
        return {"type": "BasePresent", "team": serialize_team(contents.team)}
    elif isinstance(contents, FoodPresent):
        return {"type": "FoodPresent", "count": contents.count}
    else:
        raise ValueError(f"Unknown cell contents type: {type(contents)}")


def deserialize_cell_contents(data: Dict[str, Any]) -> CellContents:
    if data["type"] == "Empty":
        return Empty()
    elif data["type"] == "UnitPresent":
        return UnitPresent(
            team=deserialize_team(data["team"]), unit_id=UnitId(data["unit_id"])
        )
    elif data["type"] == "BasePresent":
        return BasePresent(team=deserialize_team(data["team"]))
    elif data["type"] == "FoodPresent":
        return FoodPresent(count=data["count"])
    else:
        raise ValueError(f"Unknown cell contents type: {data['type']}")


# ===== Orders =====


def serialize_order(order: Order) -> Dict[str, Any]:
    if isinstance(order, Move):
        return {"type": "Move", "target": serialize_pos(order.target)}
    else:
        raise ValueError(f"Unknown order type: {type(order)}")


def deserialize_order(data: Dict[str, Any]) -> Order:
    if data["type"] == "Move":
        return Move(target=deserialize_pos(data["target"]))
    else:
        raise ValueError(f"Unknown order type: {data['type']}")


# ===== Conditions =====


def serialize_condition(condition: Any) -> Dict[str, Any]:
    if isinstance(condition, EnemyInRangeCondition):
        return {"type": "EnemyInRangeCondition", "distance": condition.distance}
    elif isinstance(condition, BaseVisibleCondition):
        return {"type": "BaseVisibleCondition"}
    elif isinstance(condition, PositionReachedCondition):
        return {
            "type": "PositionReachedCondition",
            "position": serialize_pos(condition.position),
        }
    elif isinstance(condition, FoodInRangeCondition):
        return {"type": "FoodInRangeCondition", "distance": condition.distance}
    else:
        raise ValueError(f"Unknown condition type: {type(condition)}")


def deserialize_condition(data: Dict[str, Any]) -> Any:
    if data["type"] == "EnemyInRangeCondition":
        return EnemyInRangeCondition(distance=data["distance"])
    elif data["type"] == "BaseVisibleCondition":
        return BaseVisibleCondition()
    elif data["type"] == "PositionReachedCondition":
        return PositionReachedCondition(position=deserialize_pos(data["position"]))
    elif data["type"] == "FoodInRangeCondition":
        return FoodInRangeCondition(distance=data["distance"])
    else:
        raise ValueError(f"Unknown condition type: {data['type']}")


# ===== Actions =====


def serialize_action(action: Any) -> Dict[str, Any]:
    if isinstance(action, MoveThereAction):
        return {"type": "MoveThereAction"}
    elif isinstance(action, MoveHomeAction):
        return {"type": "MoveHomeAction"}
    else:
        raise ValueError(f"Unknown action type: {type(action)}")


def deserialize_action(data: Dict[str, Any]) -> Any:
    if data["type"] == "MoveThereAction":
        return MoveThereAction()
    elif data["type"] == "MoveHomeAction":
        return MoveHomeAction()
    else:
        raise ValueError(f"Unknown action type: {data['type']}")


# ===== Interrupts =====


def serialize_interrupt(interrupt: Interrupt[Any]) -> Dict[str, Any]:
    return {
        "condition": serialize_condition(interrupt.condition),
        "actions": [serialize_action(action) for action in interrupt.actions],
    }


def deserialize_interrupt(data: Dict[str, Any]) -> Interrupt[Any]:
    return Interrupt(
        condition=deserialize_condition(data["condition"]),
        actions=[deserialize_action(action) for action in data["actions"]],
    )


# ===== Plan =====


def serialize_plan(plan: Plan) -> Dict[str, Any]:
    return {
        "orders": [serialize_order(order) for order in plan.orders],
        "interrupts": [serialize_interrupt(interrupt) for interrupt in plan.interrupts],
    }


def deserialize_plan(data: Dict[str, Any]) -> Plan:
    return Plan(
        orders=[deserialize_order(order) for order in data["orders"]],
        interrupts=[deserialize_interrupt(interrupt) for interrupt in data["interrupts"]],
    )


# ===== Unit =====


def serialize_unit(unit: Unit) -> Dict[str, Any]:
    return {
        "id": int(unit.id),
        "team": serialize_team(unit.team),
        "pos": serialize_pos(unit.pos),
        "original_pos": serialize_pos(unit.original_pos),
        "carrying_food": unit.carrying_food,
        "visibility_radius": unit.visibility_radius,
        "plan": serialize_plan(unit.plan),
        # Note: We don't serialize observation_log and last_sync_tick
        # as they're not needed for player knowledge
    }


def deserialize_unit(data: Dict[str, Any]) -> Unit:
    return Unit(
        id=UnitId(data["id"]),
        team=deserialize_team(data["team"]),
        pos=deserialize_pos(data["pos"]),
        original_pos=deserialize_pos(data["original_pos"]),
        carrying_food=data["carrying_food"],
        visibility_radius=data["visibility_radius"],
        plan=deserialize_plan(data["plan"]),
        observation_log={},  # Empty in deserialized units
        last_sync_tick=Timestamp(0),
    )


# ===== Observation Log =====


def serialize_observation_log(log: ObservationLog) -> Dict[str, Any]:
    """Serialize an observation log (dict[Timestamp, dict[Pos, list[CellContents]]])."""
    return {
        str(timestamp): {
            f"{pos.x},{pos.y}": [serialize_cell_contents(c) for c in contents_list]
            for pos, contents_list in observations.items()
        }
        for timestamp, observations in log.items()
    }


def deserialize_observation_log(data: Dict[str, Any]) -> ObservationLog:
    """Deserialize an observation log."""
    result: ObservationLog = {}
    for timestamp_str, observations in data.items():
        timestamp = Timestamp(int(timestamp_str))
        result[timestamp] = {}
        for pos_str, contents_list in observations.items():
            x, y = pos_str.split(",")
            pos = Pos(int(x), int(y))
            result[timestamp][pos] = [deserialize_cell_contents(c) for c in contents_list]
    return result


# ===== Expected Trajectory =====


def serialize_expected_trajectory(trajectory: ExpectedTrajectory) -> Dict[str, Any]:
    return {
        "unit_id": int(trajectory.unit_id),
        "start_tick": trajectory.start_tick,
        "positions": [serialize_pos(pos) for pos in trajectory.positions],
    }


def deserialize_expected_trajectory(data: Dict[str, Any]) -> ExpectedTrajectory:
    return ExpectedTrajectory(
        unit_id=UnitId(data["unit_id"]),
        start_tick=Timestamp(data["start_tick"]),
        positions=[deserialize_pos(pos) for pos in data["positions"]],
    )


# ===== Player Knowledge =====


def serialize_player_knowledge(knowledge: PlayerKnowledge) -> Dict[str, Any]:
    return {
        "team": serialize_team(knowledge.team),
        "grid_width": knowledge.grid_width,
        "grid_height": knowledge.grid_height,
        "tick": knowledge.tick,
        "all_observations": serialize_observation_log(knowledge.all_observations),
        "last_seen": {
            int(unit_id): {
                "timestamp": timestamp,
                "unit": serialize_unit(unit),
            }
            for unit_id, (timestamp, unit) in knowledge.last_seen.items()
        },
        "expected_trajectories": {
            int(unit_id): serialize_expected_trajectory(trajectory)
            for unit_id, trajectory in knowledge.expected_trajectories.items()
        },
        "last_observations": {
            f"{pos.x},{pos.y}": {
                "timestamp": timestamp,
                "contents": [serialize_cell_contents(c) for c in contents_list],
            }
            for pos, (timestamp, contents_list) in knowledge.last_observations.items()
        },
    }


def deserialize_player_knowledge(data: Dict[str, Any]) -> PlayerKnowledge:
    knowledge = PlayerKnowledge(
        team=deserialize_team(data["team"]),
        grid_width=data["grid_width"],
        grid_height=data["grid_height"],
        tick=Timestamp(data["tick"]),
    )

    knowledge.all_observations = deserialize_observation_log(data["all_observations"])

    knowledge.last_seen = {
        UnitId(int(unit_id_str)): (
            Timestamp(value["timestamp"]),
            deserialize_unit(value["unit"]),
        )
        for unit_id_str, value in data["last_seen"].items()
    }

    knowledge.expected_trajectories = {
        UnitId(int(unit_id_str)): deserialize_expected_trajectory(trajectory)
        for unit_id_str, trajectory in data["expected_trajectories"].items()
    }

    knowledge.last_observations = {}
    for pos_str, value in data["last_observations"].items():
        x, y = pos_str.split(",")
        pos = Pos(int(x), int(y))
        knowledge.last_observations[pos] = (
            Timestamp(value["timestamp"]),
            [deserialize_cell_contents(c) for c in value["contents"]],
        )

    return knowledge
