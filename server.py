"""HTTP server for multiplayer game mode."""

from __future__ import annotations
import time
from typing import Any
from flask import Flask, request, jsonify

from core import Pos
from knowledge import PlayerKnowledge
from mechanics import (
    GameState,
    Plan,
    Team,
    UnitId,
    Move,
    Interrupt,
    EnemyInRangeCondition,
    FoodInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    MoveHomeAction,
    MoveThereAction,
    Empty,
    UnitPresent,
    BasePresent,
    FoodPresent,
)


def create_server(
    state: GameState, knowledge: dict[Team, PlayerKnowledge], port: int = 5000
) -> Flask:
    """Create Flask server for multiplayer game."""
    app = Flask(__name__)

    @app.route("/knowledge/<team_name>", methods=["GET"])
    def get_knowledge(team_name: str) -> Any:
        """Wait for knowledge to reach tick>=t, then return serialized knowledge."""
        try:
            team = Team(team_name)
        except ValueError:
            return jsonify({"error": f"Invalid team: {team_name}"}), 400

        # Get tick parameter
        tick_param = request.args.get("tick")
        if tick_param is None:
            return jsonify({"error": "Missing tick parameter"}), 400

        try:
            target_tick = int(tick_param)
        except ValueError:
            return jsonify({"error": "Invalid tick parameter"}), 400

        # Wait for knowledge to reach target tick
        timeout = 30  # 30 second timeout
        start_time = time.time()
        while knowledge[team].tick < target_tick:
            if time.time() - start_time > timeout:
                return jsonify({"error": "Timeout waiting for tick"}), 408
            time.sleep(0.05)

        # Serialize and return knowledge
        return jsonify(serialize_knowledge(knowledge[team]))

    @app.route("/act/<team_name>", methods=["POST"])
    def set_plan(team_name: str) -> Any:
        """Set unit's plan if it's in the base."""
        try:
            team = Team(team_name)
        except ValueError:
            return jsonify({"error": f"Invalid team: {team_name}"}), 400

        data = request.get_json()
        if data is None:
            return jsonify({"error": "Missing JSON body"}), 400

        unit_id = data.get("unit_id")
        plan_data = data.get("plan")

        if unit_id is None or plan_data is None:
            return jsonify({"error": "Missing unit_id or plan"}), 400

        # Find the unit
        unit = None
        for u in state.units:
            if u.id == UnitId(unit_id) and u.team == team:
                unit = u
                break

        if unit is None:
            return jsonify({"success": False, "error": "Unit not found"}), 404

        # Check if unit is in base
        if not unit.is_in_base(state):
            return jsonify({"success": False, "error": "Unit not in base"}), 400

        # Deserialize and set the plan
        try:
            plan = deserialize_plan(plan_data)
            unit.plan = plan
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    return app


def serialize_knowledge(knowledge: PlayerKnowledge) -> dict[str, Any]:
    """Serialize PlayerKnowledge to JSON-compatible dict."""
    # Serialize all_observations
    all_observations = {}
    for tick, obs_dict in knowledge.all_observations.items():
        observations = {}
        for pos, contents_list in obs_dict.items():
            pos_str = f"{pos.x},{pos.y}"
            observations[pos_str] = [serialize_cell_contents(c) for c in contents_list]
        all_observations[str(tick)] = observations

    # Serialize last_observations
    last_observations = {}
    for pos, (tick, contents_list) in knowledge.last_observations.items():
        pos_str = f"{pos.x},{pos.y}"
        last_observations[pos_str] = [
            tick,
            [serialize_cell_contents(c) for c in contents_list],
        ]

    # Serialize units_in_base
    units_in_base = [serialize_unit(unit) for unit in knowledge.units_in_base]

    return {
        "team": knowledge.team.value,
        "grid_width": knowledge.grid_width,
        "grid_height": knowledge.grid_height,
        "tick": knowledge.tick,
        "all_observations": all_observations,
        "last_observations": last_observations,
        "units_in_base": units_in_base,
    }


def serialize_cell_contents(contents: Any) -> dict[str, Any]:
    """Serialize CellContents to JSON-compatible dict."""
    if isinstance(contents, Empty):
        return {"type": "Empty"}
    elif isinstance(contents, UnitPresent):
        return {"type": "UnitPresent", "team": contents.team.value, "unit_id": contents.unit_id}
    elif isinstance(contents, BasePresent):
        return {"type": "BasePresent", "team": contents.team.value}
    elif isinstance(contents, FoodPresent):
        return {"type": "FoodPresent", "count": contents.count}
    else:
        raise ValueError(f"Unknown contents type: {type(contents)}")


def serialize_unit(unit: Any) -> dict[str, Any]:
    """Serialize a Unit to JSON-compatible dict."""
    from mechanics import Unit, Plan

    return {
        "id": unit.id,
        "team": unit.team.value,
        "pos": {"x": unit.pos.x, "y": unit.pos.y},
        "original_pos": {"x": unit.original_pos.x, "y": unit.original_pos.y},
        "plan": serialize_plan(unit.plan),
        "visibility_radius": unit.visibility_radius,
        "carrying_food": unit.carrying_food,
    }


def serialize_plan(plan: Any) -> dict[str, Any]:
    """Serialize a Plan to JSON-compatible dict."""
    from mechanics import Move

    orders = []
    for order in plan.orders:
        if isinstance(order, Move):
            orders.append({
                "type": "Move",
                "target": {"x": order.target.x, "y": order.target.y},
            })

    # Note: We're not serializing interrupts here as they're complex
    # and clients can use default interrupts
    return {
        "orders": orders,
        "interrupts": [],
    }


def deserialize_plan(data: dict[str, Any]) -> Plan:
    """Deserialize a plan from JSON data."""
    orders = [deserialize_order(o) for o in data.get("orders", [])]
    interrupts = [deserialize_interrupt(i) for i in data.get("interrupts", [])]
    return Plan(orders=orders, interrupts=interrupts)


def deserialize_order(data: dict[str, Any]) -> Any:
    """Deserialize an order from JSON data."""
    order_type = data.get("type")
    if order_type == "Move":
        target = data["target"]
        return Move(target=Pos(target["x"], target["y"]))
    else:
        raise ValueError(f"Unknown order type: {order_type}")


def deserialize_interrupt(data: dict[str, Any]) -> Interrupt[Any]:
    """Deserialize an interrupt from JSON data."""
    # Deserialize condition
    cond_data = data["condition"]
    cond_type = cond_data["type"]
    condition: Any
    if cond_type == "EnemyInRange":
        condition = EnemyInRangeCondition(distance=cond_data["distance"])
    elif cond_type == "FoodInRange":
        condition = FoodInRangeCondition(distance=cond_data["distance"])
    elif cond_type == "BaseVisible":
        condition = BaseVisibleCondition()
    elif cond_type == "PositionReached":
        pos_data = cond_data["position"]
        condition = PositionReachedCondition(position=Pos(pos_data["x"], pos_data["y"]))
    else:
        raise ValueError(f"Unknown condition type: {cond_type}")

    # Deserialize actions
    actions: list[Any] = []
    for action_data in data["actions"]:
        action_type = action_data["type"]
        if action_type == "MoveHome":
            actions.append(MoveHomeAction())
        elif action_type == "MoveThere":
            actions.append(MoveThereAction())
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    return Interrupt(condition=condition, actions=actions)
