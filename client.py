"""Client abstraction for interacting with game state (local or remote)."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, cast, TYPE_CHECKING
import json
import requests
import time

from core import Pos, Region, Timestamp
from knowledge import PlayerKnowledge
from mechanics import GameState, PlayerAction, Team, UnitId, UnitType, Unit
from planning import Plan, PlanningMind

if TYPE_CHECKING:
    from server import ObservationStore


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
    def add_player_action(self, team: Team, action: PlayerAction) -> None:
        pass

    @abstractmethod
    def get_food_count_in_base(self, team: Team) -> int:
        """Get the total amount of food in a team's base."""
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
    observation_store: Optional["ObservationStore"] = None
    queued_player_actions: dict[Team, list[PlayerAction]] = field(default_factory=dict)

    def get_player_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        while self.state.now < tick:
            time.sleep(0.03)
        return self.knowledge[team]

    def add_player_action(self, team: Team, action: PlayerAction) -> None:
        self.queued_player_actions.setdefault(team, []).append(action)

    def get_food_count_in_base(self, team: Team) -> int:
        return self.state.get_food_count_in_base(team)

    def get_base_region(self, team: Team) -> Region:
        return self.state.base_regions[team]

    def get_current_tick(self) -> Timestamp:
        return self.state.now

    def get_god_view(self) -> Optional[GameState]:
        return self.state

    def get_available_teams(self) -> list[Team]:
        return list(Team)

    def flush_queued_actions(self) -> None:
        self.state.apply_player_actions(self.queued_player_actions)
        self.queued_player_actions.clear()
        self._let_all_players_observe()

    def tick_game(self) -> None:
        self.state.tick()
        self._let_all_players_observe()

    def _let_all_players_observe(self) -> None:
        for team, knowledge in self.knowledge.items():
            units_in_base = [
                u for u in self.state.units.values()
                if u.team == team and u.is_in_base(self.state)
            ]
            # Record to observation store if available (for server mode)
            if self.observation_store is not None:
                self.observation_store.record(team, self.state.now, units_in_base)
            knowledge.observe(self.state.now, units_in_base)

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
        self._fetch_observations(after=Timestamp(-1))

    def _fetch_observations(self, after: Timestamp) -> None:
        """Fetch observations from server and apply them to local knowledge.

        Args:
            after: Fetch observations with timestamp > after
        """
        from serialization import deserialize_unit

        print("fetching observations", self.team.name, "after", after)
        response = requests.get(
            f"{self.url}/observations/{self.team.name}",
            params={"after": after},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # On first fetch, initialize knowledge with grid dimensions
        if self._current_knowledge is None:
            self._current_knowledge = PlayerKnowledge(
                team=self.team,
                grid_width=data["grid_width"],
                grid_height=data["grid_height"],
                tick=Timestamp(0),
            )
            self._base_region = deserialize_region(data["base_region"])

        # Deserialize and apply observations in timestamp order
        observations: dict[Timestamp, list[Unit]] = {}
        for tick_str, units_data in data.get("observations", {}).items():
            tick = Timestamp(int(tick_str))
            observations[tick] = [deserialize_unit(u) for u in units_data]

        # Apply observations in order
        for tick in sorted(observations.keys()):
            self._current_knowledge.observe(tick, observations[tick])
            self._last_tick = tick

        print(
            "fetched observations for",
            self.team.name,
            "after",
            after,
            ", now t=",
            self._last_tick,
        )

    def get_player_knowledge(self, team: Team, tick: Timestamp) -> PlayerKnowledge:
        if team != self.team:
            raise ValueError(f"Can only view own team ({self.team}) in remote mode")

        if tick <= self._last_tick and self._current_knowledge is not None:
            return self._current_knowledge

        # Request observations after our current tick
        self._fetch_observations(after=self._last_tick)

        if self._current_knowledge is None:
            raise RuntimeError("Failed to fetch observations from server")

        return self._current_knowledge

    def add_player_action(self, team: Team, action: PlayerAction) -> None:
        if team != self.team:
            raise ValueError(f"Can only control own team ({self.team}) in remote mode")

        response = requests.post(
            f"{self.url}/act/{team.name}",
            json=serialize_player_action(action),
            timeout=5,
        )
        response.raise_for_status()

    def get_food_count_in_base(self, team: Team) -> int:
        if team != self.team:
            raise ValueError(f"Can only view own team ({self.team}) in remote mode")

        # Food count is included in knowledge updates
        # For now, return 0 - this will need server support
        # In a real implementation, we'd track this in _current_knowledge
        try:
            response = requests.get(
                f"{self.url}/food_count/{team.name}",
                timeout=5,
            )
            response.raise_for_status()
            return int(response.json().get("count", 0))
        except Exception as e:
            print(f"Error getting food count: {e}")
            return 0

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
