"""Game lifecycle management - initialization, ticking, and mode-specific setup."""

from __future__ import annotations
import argparse
import random
import threading
import time
from dataclasses import dataclass, field

import numpy

from client import GameClient, LocalClient, RemoteClient
from core import Timestamp
from knowledge import PlayerKnowledge
from mechanics import FoodConfig, GameState, Team, make_game
from planning import PlanningMind


@dataclass
class GameLifecycle:
    """Manages game state and background ticking/fetching."""

    client: GameClient
    knowledge: dict[Team, PlayerKnowledge]
    grid_width: int
    grid_height: int
    available_teams: list[Team]
    state: GameState | None = None  # Only available in local/server mode

    _running: bool = field(default=False, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def stop(self) -> None:
        """Stop the background game loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def run_local_game(
    grid_width: int,
    grid_height: int,
    food_config: FoodConfig,
    seconds_per_tick: float,
) -> GameLifecycle:
    """Create and start a local game."""
    state = make_game(
        make_mind=PlanningMind,
        grid_width=grid_width,
        grid_height=grid_height,
        food_config=food_config,
    )

    knowledge = {
        Team.RED: PlayerKnowledge(
            team=Team.RED,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.now,
        ),
        Team.BLUE: PlayerKnowledge(
            team=Team.BLUE,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.now,
        ),
    }

    local_client = LocalClient(state=state, knowledge=knowledge)

    lifecycle = GameLifecycle(
        client=local_client,
        knowledge=knowledge,
        grid_width=grid_width,
        grid_height=grid_height,
        available_teams=list(Team),
        state=state,
    )

    def tick_loop() -> None:
        while lifecycle._running:
            local_client.flush_queued_actions()
            local_client.tick_game()
            time.sleep(seconds_per_tick)

    lifecycle._running = True
    lifecycle._thread = threading.Thread(target=tick_loop, daemon=True)
    lifecycle._thread.start()

    return lifecycle


def run_server_game(
    grid_width: int,
    grid_height: int,
    food_config: FoodConfig,
    seconds_per_tick: float,
    port: int,
) -> GameLifecycle:
    """Create and start a game server."""
    from server import GameServer

    state = make_game(
        make_mind=PlanningMind,
        grid_width=grid_width,
        grid_height=grid_height,
        food_config=food_config,
    )

    knowledge = {
        Team.RED: PlayerKnowledge(
            team=Team.RED,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.now,
        ),
        Team.BLUE: PlayerKnowledge(
            team=Team.BLUE,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.now,
        ),
    }

    local_client = LocalClient(state=state, knowledge=knowledge)

    # Start the HTTP server
    server = GameServer(state, knowledge, port=port)
    server.start()
    print(f"Server started on port {port}")

    lifecycle = GameLifecycle(
        client=local_client,
        knowledge=knowledge,
        grid_width=grid_width,
        grid_height=grid_height,
        available_teams=list(Team),
        state=state,
    )

    def tick_loop() -> None:
        while lifecycle._running:
            local_client.flush_queued_actions()
            local_client.tick_game()
            time.sleep(seconds_per_tick)

    lifecycle._running = True
    lifecycle._thread = threading.Thread(target=tick_loop, daemon=True)
    lifecycle._thread.start()

    return lifecycle


def run_client_game(
    url: str,
    team: Team,
) -> GameLifecycle:
    """Connect to a remote game server."""
    remote_client = RemoteClient(url=url, team=team)

    # Fetch initial knowledge to get grid dimensions
    initial_knowledge = remote_client.get_player_knowledge(team, Timestamp(0))
    grid_width = initial_knowledge.grid_width
    grid_height = initial_knowledge.grid_height

    # Create knowledge dict with the fetched knowledge
    # Only the client's team has real knowledge; others are dummies
    knowledge: dict[Team, PlayerKnowledge] = {}
    for t in Team:
        if t == team:
            knowledge[t] = initial_knowledge
        else:
            knowledge[t] = PlayerKnowledge(
                team=t,
                grid_width=grid_width,
                grid_height=grid_height,
                tick=Timestamp(0),
            )

    lifecycle = GameLifecycle(
        client=remote_client,
        knowledge=knowledge,
        grid_width=grid_width,
        grid_height=grid_height,
        available_teams=[team],
        state=None,
    )

    def fetch_loop() -> None:
        while lifecycle._running:
            try:
                new_knowledge = remote_client.get_player_knowledge(
                    team, remote_client.get_current_tick() + 1
                )
                lifecycle.knowledge[team] = new_knowledge
            except Exception as e:
                print(f"Error fetching knowledge: {e}")
                time.sleep(0.5)

    lifecycle._running = True
    lifecycle._thread = threading.Thread(target=fetch_loop, daemon=True)
    lifecycle._thread.start()

    return lifecycle


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for game configuration."""
    from mechanics import GRID_SIZE

    parser = argparse.ArgumentParser(description="Ant RTS Game")
    parser.add_argument(
        "--width", type=int, default=GRID_SIZE, help="Width of the grid (default: 32)"
    )
    parser.add_argument(
        "--height", type=int, default=GRID_SIZE, help="Height of the grid (default: 32)"
    )
    parser.add_argument(
        "--food-scale",
        type=float,
        default=10.0,
        help="Perlin noise scale for food generation (default: 10.0)",
    )
    parser.add_argument(
        "--food-max-prob",
        type=float,
        default=0.1,
        help="Maximum probability of food in a cell (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for food generation (default: random)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "server", "client"],
        default="local",
        help="Game mode: local (default), server, or client",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for server mode (default: 5000)",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Server URL for client mode (e.g., http://localhost:5000)",
    )
    parser.add_argument(
        "--team",
        type=str,
        choices=["RED", "BLUE"],
        help="Team to play as in client mode",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "client":
        if not args.url:
            parser.error("--url is required for client mode")
        if not args.team:
            parser.error("--team is required for client mode")

    return args


def create_lifecycle_from_args(args: argparse.Namespace) -> GameLifecycle:
    """Create a GameLifecycle based on parsed command-line arguments."""
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    seconds_per_tick = 1 / 5

    if args.mode == "client":
        return run_client_game(
            url=args.url,
            team=Team[args.team],
        )
    else:
        food_config = FoodConfig(
            scale=args.food_scale,
            max_prob=args.food_max_prob,
        )

        if args.mode == "server":
            return run_server_game(
                grid_width=args.width,
                grid_height=args.height,
                food_config=food_config,
                seconds_per_tick=seconds_per_tick,
                port=args.port,
            )
        else:
            return run_local_game(
                grid_width=args.width,
                grid_height=args.height,
                food_config=food_config,
                seconds_per_tick=seconds_per_tick,
            )
