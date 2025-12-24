"""Client abstraction for interacting with game state (local or remote)."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import json
import requests
import time

from core import Pos, Region, Timestamp
from mechanics import Team, GameState, Plan, UnitId
from knowledge import PlayerKnowledge


class GameClient(ABC):
    """Abstract interface for interacting with game state.

    This abstraction allows the UI to work with both local and remote game states
    without needing conditionals everywhere about what mode we're in.
    """

    @abstractmethod
    def get_player_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        """Wait until at least the given time, then get the current knowledge for a team."""
        pass

    @abstractmethod
    def set_unit_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> None:
        """Set a unit's plan (if it's in that team's base)."""
        pass

    @abstractmethod
    def get_base_region(self, team: Team) -> Region:
        """Get the base region for a team."""
        pass

    @abstractmethod
    def get_current_tick(self) -> Timestamp:
        """Get the current game tick."""
        pass

    @abstractmethod
    def get_god_view(self) -> Optional[GameState]:
        """Get the full game state (god's eye view).

        Returns None in client mode, GameState in local/server mode.
        """
        pass

    @abstractmethod
    def get_available_teams(self) -> list[Team]:
        """Get list of teams that can be viewed/controlled.

        Returns all teams in local/server mode, only client's team in client mode.
        """
        pass


@dataclass
class LocalClient(GameClient):
    """Client for local or server mode - has direct access to GameState."""

    state: GameState
    knowledge: dict[Team, PlayerKnowledge]

    def get_player_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        while self.state.tick < tick:
            time.sleep(0.03) # TODO: use a Condition instead of polling
        return self.knowledge[team]

    def set_unit_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> None:
        # LocalClient has direct access, so just call the method
        self.state.set_unit_plan(unit_id, plan)

    def get_base_region(self, team: Team) -> Region:
        return self.state.base_regions[team]

    def get_current_tick(self) -> Timestamp:
        return self.state.tick

    def get_god_view(self) -> Optional[GameState]:
        return self.state

    def get_available_teams(self) -> list[Team]:
        return list(Team)


@dataclass
class RemoteClient(GameClient):
    """Client for remote mode - communicates with server via HTTP."""

    url: str
    team: Team
    _current_knowledge: Optional[PlayerKnowledge] = None
    _base_region: Optional[Region] = None
    _last_tick: Timestamp = Timestamp(0)

    def __post_init__(self) -> None:
        """Initialize by fetching initial state."""
        self._fetch_knowledge(tick=Timestamp(0))

    def _fetch_knowledge(self, tick: Timestamp) -> None:
        """Fetch knowledge from server, waiting for next tick if needed."""
        # Wait for tick > last_tick
        print('fetching knowledge', self.team.name, tick)
        response = requests.get(
            f"{self.url}/knowledge/{self.team.name}",
            params={"tick": tick},
            timeout=30
        )
        # print('response', response.status_code, response.text)
        response.raise_for_status()
        data = response.json()

        # Deserialize knowledge
        from serialization import deserialize_player_knowledge
        self._current_knowledge = deserialize_player_knowledge(data["knowledge"])
        self._base_region = deserialize_region(data["base_region"])
        self._last_tick = self._current_knowledge.tick
        print('fetched knowledge for', self.team.name, 'at >=', tick, ', now t=', self._last_tick)

    def get_player_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        if team != self.team:
            raise ValueError(f"Can only view own team ({self.team}) in remote mode")

        if tick <= self._last_tick and self._current_knowledge is not None:
            return self._current_knowledge

        self._fetch_knowledge(tick=tick)

        if self._current_knowledge is None:
            raise RuntimeError("Failed to fetch knowledge from server")

        return self._current_knowledge

    def set_unit_plan(self, team: Team, unit_id: UnitId, plan: Plan) -> None:
        if team != self.team:
            raise ValueError(f"Can only control own team ({self.team}) in remote mode")

        from serialization import serialize_plan
        try:
            response = requests.post(
                f"{self.url}/act/{team.name}/{unit_id}",
                json=serialize_plan(plan),
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Error setting unit plan: {e}")
            raise

    def get_base_region(self, team: Team) -> Region:
        if team != self.team:
            raise ValueError(f"Can only view own team ({self.team}) in remote mode")

        if self._base_region is None:
            raise RuntimeError("Base region not yet fetched from server")

        return self._base_region

    def get_current_tick(self) -> Timestamp:
        if self._current_knowledge is None:
            raise RuntimeError("Knowledge not yet fetched from server")
        return self._current_knowledge.tick

    def get_god_view(self) -> Optional[GameState]:
        # Never available in client mode
        return None

    def get_available_teams(self) -> list[Team]:
        # Can only view own team in client mode
        return [self.team]


def deserialize_region(data: dict[str, Any]) -> Region:
    """Deserialize a Region from JSON."""
    cells = frozenset(Pos(x=p["x"], y=p["y"]) for p in data["cells"])
    return Region(cells=cells)
