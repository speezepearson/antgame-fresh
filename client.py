"""Game client interface and implementations for multiplayer support."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import time
from typing import Any
import requests

from core import Pos, Timestamp
from knowledge import PlayerKnowledge
from mechanics import GameState, Plan, Team, UnitId


class GameClient(ABC):
    """Interface for interacting with game state (local or remote)."""

    @abstractmethod
    def get_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        """Wait for team's knowledge to reach tick>=t, then return it."""
        ...

    @abstractmethod
    def set_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> bool:
        """Set unit's plan if it's in the base. Returns True if successful."""
        ...


class LocalGameClient(GameClient):
    """Game client that directly interacts with local GameState."""

    def __init__(self, state: GameState, knowledge: dict[Team, PlayerKnowledge]):
        self.state = state
        self.knowledge = knowledge

    def get_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        """Wait for team's knowledge to reach tick>=t, then return it."""
        # Wait until knowledge reaches the desired tick
        while self.knowledge[team].tick < tick:
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        return self.knowledge[team]

    def set_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> bool:
        """Set unit's plan if it's in the base. Returns True if successful."""
        # Find the unit
        unit = None
        for u in self.state.units:
            if u.id == unit_id and u.team == team:
                unit = u
                break

        if unit is None:
            return False

        # Check if unit is in base
        if not unit.is_in_base(self.state):
            return False

        # Set the plan
        unit.plan = plan
        return True


class RemoteGameClient(GameClient):
    """Game client that interacts with remote server via HTTP."""

    def __init__(self, server_url: str, team: Team):
        self.server_url = server_url.rstrip('/')
        self.team = team

    def get_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        """Wait for team's knowledge to reach tick>=t, then return it."""
        url = f"{self.server_url}/knowledge/{team.value}?tick={tick}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        return self._deserialize_knowledge(data)

    def set_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> bool:
        """Set unit's plan if it's in the base. Returns True if successful."""
        url = f"{self.server_url}/act/{team.value}"
        payload = {
            "unit_id": unit_id,
            "plan": self._serialize_plan(plan),
        }
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        return bool(result.get("success", False))

    def _serialize_plan(self, plan: Plan) -> dict[str, Any]:
        """Serialize a plan to JSON-compatible dict."""
        return {
            "orders": [self._serialize_order(o) for o in plan.orders],
            "interrupts": [self._serialize_interrupt(i) for i in plan.interrupts],
        }

    def _serialize_order(self, order: Any) -> dict[str, Any]:
        """Serialize an order to JSON-compatible dict."""
        # Import here to avoid circular imports
        from mechanics import Move

        if isinstance(order, Move):
            return {
                "type": "Move",
                "target": {"x": order.target.x, "y": order.target.y},
            }
        else:
            raise ValueError(f"Unknown order type: {type(order)}")

    def _serialize_interrupt(self, interrupt: Any) -> dict[str, Any]:
        """Serialize an interrupt to JSON-compatible dict."""
        from mechanics import (
            EnemyInRangeCondition,
            FoodInRangeCondition,
            BaseVisibleCondition,
            PositionReachedCondition,
            MoveHomeAction,
            MoveThereAction,
        )

        # Serialize condition
        condition = interrupt.condition
        if isinstance(condition, EnemyInRangeCondition):
            cond_dict = {"type": "EnemyInRange", "distance": condition.distance}
        elif isinstance(condition, FoodInRangeCondition):
            cond_dict = {"type": "FoodInRange", "distance": condition.distance}
        elif isinstance(condition, BaseVisibleCondition):
            cond_dict = {"type": "BaseVisible"}
        elif isinstance(condition, PositionReachedCondition):
            cond_dict = {"type": "PositionReached", "position": {"x": condition.position.x, "y": condition.position.y}}
        else:
            raise ValueError(f"Unknown condition type: {type(condition)}")

        # Serialize actions
        actions = []
        for action in interrupt.actions:
            if isinstance(action, MoveHomeAction):
                actions.append({"type": "MoveHome"})
            elif isinstance(action, MoveThereAction):
                actions.append({"type": "MoveThere"})
            else:
                raise ValueError(f"Unknown action type: {type(action)}")

        return {
            "condition": cond_dict,
            "actions": actions,
        }

    def _deserialize_knowledge(self, data: dict[str, Any]) -> PlayerKnowledge:
        """Deserialize PlayerKnowledge from JSON data."""
        from mechanics import UnitPresent, BasePresent, FoodPresent, Empty, CellContents

        # Deserialize all_observations
        all_observations: dict[Timestamp, dict[Pos, list[CellContents]]] = {}
        for tick_str, obs_dict in data.get("all_observations", {}).items():
            tick = int(tick_str)
            observations: dict[Pos, list[CellContents]] = {}
            for pos_str, contents_list in obs_dict.items():
                x, y = map(int, pos_str.split(","))
                pos = Pos(x, y)
                contents: list[CellContents] = []
                for c in contents_list:
                    if c["type"] == "Empty":
                        contents.append(Empty())
                    elif c["type"] == "UnitPresent":
                        contents.append(UnitPresent(Team(c["team"]), UnitId(c["unit_id"])))
                    elif c["type"] == "BasePresent":
                        contents.append(BasePresent(Team(c["team"])))
                    elif c["type"] == "FoodPresent":
                        contents.append(FoodPresent(c["count"]))
                observations[pos] = contents
            all_observations[tick] = observations

        # Deserialize last_observations
        last_observations: dict[Pos, tuple[Timestamp, list[CellContents]]] = {}
        for pos_str, (tick, contents_list) in data.get("last_observations", {}).items():
            x, y = map(int, pos_str.split(","))
            pos = Pos(x, y)
            last_contents: list[CellContents] = []
            for c in contents_list:
                if c["type"] == "Empty":
                    last_contents.append(Empty())
                elif c["type"] == "UnitPresent":
                    last_contents.append(UnitPresent(Team(c["team"]), UnitId(c["unit_id"])))
                elif c["type"] == "BasePresent":
                    last_contents.append(BasePresent(Team(c["team"])))
                elif c["type"] == "FoodPresent":
                    last_contents.append(FoodPresent(c["count"]))
            last_observations[pos] = (tick, last_contents)

        # Deserialize units_in_base
        units_in_base = [self._deserialize_unit(u) for u in data.get("units_in_base", [])]

        return PlayerKnowledge(
            team=Team(data["team"]),
            grid_width=data["grid_width"],
            grid_height=data["grid_height"],
            tick=data["tick"],
            all_observations=all_observations,
            last_observations=last_observations,
            units_in_base=units_in_base,
        )

    def _deserialize_unit(self, data: dict[str, Any]) -> Any:
        """Deserialize a Unit from JSON data."""
        from mechanics import Unit, UnitId, Plan, Move, Order

        # Deserialize plan
        orders: list[Order] = []
        for order_data in data.get("plan", {}).get("orders", []):
            if order_data["type"] == "Move":
                target = order_data["target"]
                orders.append(Move(target=Pos(target["x"], target["y"])))

        plan = Plan(orders=orders, interrupts=[])

        return Unit(
            id=UnitId(data["id"]),
            team=Team(data["team"]),
            pos=Pos(data["pos"]["x"], data["pos"]["y"]),
            original_pos=Pos(data["original_pos"]["x"], data["original_pos"]["y"]),
            plan=plan,
            visibility_radius=data.get("visibility_radius", 5),
            carrying_food=data.get("carrying_food", 0),
        )
