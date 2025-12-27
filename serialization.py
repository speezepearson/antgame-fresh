"""Serialization and deserialization for network communication."""

from __future__ import annotations
from typing import Any, Dict, List, cast
import json

from core import Pos, Region, Timestamp
from mechanics import (
    Mind,
    Team,
    UnitId,
    Unit,
    UnitType,
    Empty,
    UnitPresent,
    BasePresent,
    FoodPresent,
    CellContents,
)
from planning import (
    Plan,
    Move,
    Order,
    Interrupt,
    EnemyInRangeCondition,
    IdleCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodInRangeCondition,
    MoveThereAction,
    MoveHomeAction,
    ResumeAction,
    PlanningMind,
)
from logbook import Logbook, ObservationLog
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


def serialize_unit_type(unit_type: UnitType) -> str:
    return unit_type.name


def deserialize_unit_type(data: str) -> UnitType:
    return UnitType[data]


# ===== Cell Contents =====


def serialize_cell_contents(contents: CellContents) -> Dict[str, Any]:
    if isinstance(contents, Empty):
        return {"type": "Empty"}
    elif isinstance(contents, UnitPresent):
        return {
            "type": "UnitPresent",
            "team": serialize_team(contents.team),
            "unit_id": int(contents.unit_id),
            "unit_type": serialize_unit_type(contents.unit_type),
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
            team=deserialize_team(data["team"]),
            unit_id=UnitId(data["unit_id"]),
            unit_type=deserialize_unit_type(data.get("unit_type", "FIGHTER")),
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
    elif isinstance(condition, IdleCondition):
        return {"type": "IdleCondition"}
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
    elif data["type"] == "IdleCondition":
        return IdleCondition()
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
    elif isinstance(action, ResumeAction):
        return {"type": "ResumeAction"}
    else:
        raise ValueError(f"Unknown action type: {type(action)}")


def deserialize_action(data: Dict[str, Any]) -> Any:
    if data["type"] == "MoveThereAction":
        return MoveThereAction()
    elif data["type"] == "MoveHomeAction":
        return MoveHomeAction()
    elif data["type"] == "ResumeAction":
        return ResumeAction()
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
        interrupts=[
            deserialize_interrupt(interrupt) for interrupt in data["interrupts"]
        ],
    )


# ===== Unit =====


def serialize_unit(unit: Unit) -> Dict[str, Any]:
    """Serialize a Unit. If it has a PlanningMind, also serialize plan and original_pos."""
    result: Dict[str, Any] = {
        "id": int(unit.id),
        "team": serialize_team(unit.team),
        "pos": serialize_pos(unit.pos),
        "unit_type": serialize_unit_type(unit.unit_type),
        "carrying_food": unit.carrying_food,
        "mind": serialize_mind(unit.mind),
    }
    return result

def deserialize_unit(data: Dict[str, Any]) -> Unit:
    """Deserialize a Unit. Creates a PlanningMind if plan data is present."""
    return Unit(
        id=UnitId(data["id"]),
        team=deserialize_team(data["team"]),
        mind=deserialize_mind(data["mind"]),
        pos=deserialize_pos(data["pos"]),
        unit_type=deserialize_unit_type(data.get("unit_type", "FIGHTER")),
        carrying_food=data["carrying_food"],
    )


# ===== Mind =====

def serialize_mind(mind: Mind) -> Dict[str, Any]:
    if isinstance(mind, PlanningMind):
        return {
            "type": "PlanningMind",
            "logbook": serialize_logbook(mind.logbook),
            "plan": serialize_plan(mind.plan),
            "original_pos": serialize_pos(mind.original_pos) if mind.original_pos is not None else None,
        }
    else:
        raise TypeError(f"Don't know how to serialize mind type: {type(mind)}")

def deserialize_mind(data: Dict[str, Any]) -> Mind:
    if data["type"] == "PlanningMind":
        return PlanningMind(
            logbook=deserialize_logbook(data["logbook"]),
            plan=deserialize_plan(data["plan"]),
            original_pos=deserialize_pos(data["original_pos"]) if data["original_pos"] is not None else None,
        )
    else:
        raise ValueError(f"Unknown mind type: {data['type']}")

# ===== Logbook =====


def serialize_logbook(logbook: Logbook) -> Dict[str, Any]:
    """Serialize an observation log (dict[Timestamp, dict[Pos, list[CellContents]]])."""
    return {
        "observation_log": {
            str(timestamp): {
                f"{pos.x},{pos.y}": [serialize_cell_contents(c) for c in contents_list]
                for pos, contents_list in observations.items()
            }
            for timestamp, observations in logbook.observation_log.items()
        }
    }

def deserialize_logbook(data: Dict[str, Any]) -> Logbook:
    result = Logbook()
    for timestamp_str, observations in data["observation_log"].items():
        result.add_latest_observations(now=Timestamp(int(timestamp_str)), observations={
            Pos(*[int(p) for p in pos_str.split(",")]): [deserialize_cell_contents(c) for c in contents_list]
            for pos_str, contents_list in observations.items()
        })
    return result
