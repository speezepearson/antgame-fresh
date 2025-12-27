"""HTTP server for multiplayer game (server mode)."""

from __future__ import annotations
import asyncio
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from aiohttp import web
from typing import Any, Callable

from core import Timestamp
from mechanics import Team, UnitId, Unit, GameState
from planning import PlanningMind
from knowledge import PlayerKnowledge
from serialization import (
    serialize_region,
    serialize_unit,
    deserialize_plan,
)


@dataclass
class ObservationStore:
    """Stores observation snapshots for each team at each tick.

    This is used by the server to capture and serve observations
    efficiently without serializing full PlayerKnowledge.
    """

    _observations: dict[Team, dict[Timestamp, list[Unit]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, team: Team, tick: Timestamp, units: list[Unit]) -> None:
        """Record a snapshot of observations at the given tick.

        Deep copies units since they are mutable.
        """
        with self._lock:
            if team not in self._observations:
                self._observations[team] = {}
            self._observations[team][tick] = [deepcopy(u) for u in units]

    def get_after(self, team: Team, after: Timestamp) -> dict[Timestamp, list[Unit]]:
        """Get all observations with timestamp > after."""
        with self._lock:
            team_obs = self._observations.get(team, {})
            return {tick: units for tick, units in team_obs.items() if tick > after}

    def get_latest_tick(self, team: Team) -> Timestamp:
        """Get the latest tick with observations for a team."""
        with self._lock:
            team_obs = self._observations.get(team, {})
            if not team_obs:
                return Timestamp(0)
            return max(team_obs.keys())

    def discard_up_to(self, team: Team, up_to: Timestamp) -> None:
        """Discard observations with timestamp <= up_to."""
        with self._lock:
            team_obs = self._observations.get(team, {})
            ticks_to_remove = [tick for tick in team_obs if tick <= up_to]
            for tick in ticks_to_remove:
                del team_obs[tick]


class GameServer:
    """HTTP server that exposes game state to remote clients."""

    def __init__(
        self,
        state: GameState,
        knowledge: dict[Team, PlayerKnowledge],
        observation_store: ObservationStore,
        port: int = 5000,
        ready_event: threading.Event | None = None,
    ):
        self.state = state
        self.knowledge = knowledge
        self.observation_store = observation_store
        self.port = port
        self.ready_event = ready_event
        self.server_thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _create_app(self) -> web.Application:
        """Create and configure the aiohttp application."""
        app = web.Application()
        app.router.add_get("/observations/{team_name}", self._get_observations)
        app.router.add_get("/food_count/{team_name}", self._get_food_count)
        app.router.add_post("/act/{team_name}/{unit_id}", self._set_unit_plan)

        if self.ready_event is not None:
            app.on_startup.append(self._on_startup)

        return app

    async def _on_startup(self, app: web.Application) -> None:
        """Signal that the server is ready to accept connections."""
        if self.ready_event is not None:
            self.ready_event.set()

    async def _get_observations(self, request: web.Request) -> web.Response:
        """Get observation snapshots for a team after a given tick.

        Query params:
            after: Return observations with timestamp > after (required)

        Returns:
            JSON with:
                - observations: dict mapping tick -> list of serialized units
                - base_region: the team's base region (for initial setup)
                - grid_width, grid_height: grid dimensions

        Semantics:
            - if after == state.now, returns empty observations dict
            - if after == state.now - 1, returns one batch of observations
            - After returning, discards observations at or before `after`
        """
        team_name = request.match_info["team_name"]
        try:
            team = Team[team_name]
        except KeyError:
            return web.json_response(
                {"error": f"Invalid team: {team_name}"}, status=400
            )

        # Get 'after' parameter (required)
        after_str = request.query.get("after")
        if after_str is None:
            return web.json_response(
                {"error": "Missing required 'after' query parameter"}, status=400
            )
        after = Timestamp(int(after_str))

        # Long-polling: wait until we have observations after the requested tick
        timeout = 30  # seconds
        elapsed = 0.0
        while elapsed < timeout:
            latest_tick = self.observation_store.get_latest_tick(team)
            if latest_tick > after:
                # We have new observations
                observations = self.observation_store.get_after(team, after)

                # Serialize observations: dict[Timestamp, list[Unit]] -> dict[str, list[dict]]
                serialized_observations = {
                    str(tick): [serialize_unit(u) for u in units]
                    for tick, units in observations.items()
                }

                print(f'serialized observations for {set(serialized_observations.keys())}: {serialized_observations}')

                # Discard old observations now that the client has caught up
                self.observation_store.discard_up_to(team, after)

                return web.json_response(
                    {
                        "observations": serialized_observations,
                        "base_region": serialize_region(self.state.base_regions[team]),
                        "grid_width": self.state.grid_width,
                        "grid_height": self.state.grid_height,
                    }
                )

            await asyncio.sleep(0.05)
            elapsed += 0.05

        # Timeout - return empty observations
        return web.json_response(
            {
                "observations": {},
                "base_region": serialize_region(self.state.base_regions[team]),
                "grid_width": self.state.grid_width,
                "grid_height": self.state.grid_height,
            }
        )

    async def _get_food_count(self, request: web.Request) -> web.Response:
        """Get the food count in a team's base."""
        team_name = request.match_info["team_name"]
        try:
            team = Team[team_name]
        except KeyError:
            return web.json_response(
                {"error": f"Invalid team: {team_name}"}, status=400
            )

        count = self.state.get_food_count_in_base(team)
        return web.json_response({"count": count})

    async def _set_unit_plan(self, request: web.Request) -> web.Response:
        """Set a unit's plan.

        Body: JSON-serialized Plan
        """
        team_name = request.match_info["team_name"]
        unit_id_str = request.match_info["unit_id"]

        try:
            team = Team[team_name]
        except KeyError:
            return web.json_response(
                {"error": f"Invalid team: {team_name}"}, status=400
            )

        try:
            plan_data = await request.json()
            plan = deserialize_plan(plan_data)
        except Exception as e:
            return web.json_response({"error": f"Invalid plan data: {e}"}, status=400)

        try:
            unit_id = UnitId(int(unit_id_str))
            unit = self.state.units.get(unit_id)
            if unit is None:
                return web.json_response({"error": f"Unit {unit_id} not found"}, status=400)
            if unit.team != team:
                return web.json_response({"error": f"Unit {unit_id} does not belong to team {team.name}"}, status=400)
            if not isinstance(unit.mind, PlanningMind):
                return web.json_response({"error": f"Unit {unit_id} does not have a PlanningMind"}, status=400)
            unit.mind.plan = plan
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"error": f"Failed to set plan: {e}"}, status=400)

    def start(self) -> None:
        """Start the server in a background thread."""
        if self.server_thread is not None:
            raise RuntimeError("Server already started")

        def run_server() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            app = self._create_app()
            self._runner = web.AppRunner(app)

            async def start_server() -> None:
                assert self._runner is not None
                await self._runner.setup()
                site = web.TCPSite(self._runner, "0.0.0.0", self.port)
                await site.start()

            self._loop.run_until_complete(start_server())
            self._loop.run_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to be ready before returning
        if self.ready_event is not None:
            self.ready_event.wait()

        print(f"Server started on port {self.port}")

    def stop(self) -> None:
        """Stop the server."""
        if self._loop is not None:

            async def cleanup() -> None:
                if self._runner is not None:
                    await self._runner.cleanup()

            self._loop.call_soon_threadsafe(self._loop.stop)
