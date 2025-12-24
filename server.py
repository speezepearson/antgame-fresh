"""HTTP server for multiplayer game (server mode)."""

from __future__ import annotations
import threading
import time
from flask import Flask, request, jsonify
from typing import Callable, Any

from core import Timestamp
from mechanics import Team, UnitId, GameState, Plan
from knowledge import PlayerKnowledge
from serialization import (
    serialize_player_knowledge,
    serialize_region,
    deserialize_plan,
)


class GameServer:
    """HTTP server that exposes game state to remote clients."""

    def __init__(
        self,
        state: GameState,
        knowledge: dict[Team, PlayerKnowledge],
        port: int = 5000,
        ready_event: threading.Event | None = None,
    ):
        self.state = state
        self.knowledge = knowledge
        self.port = port
        self.ready_event = ready_event
        self.app = Flask(__name__)
        self._setup_routes()
        self.server_thread: threading.Thread | None = None

    def _setup_routes(self) -> None:
        """Set up Flask routes."""

        @self.app.route("/knowledge/<team_name>", methods=["GET"])
        def get_knowledge(team_name: str) -> Any:
            """Get player knowledge for a team, waiting for a specific tick.

            Query params:
                tick: Wait until knowledge.tick >= this value before returning
            """
            try:
                team = Team[team_name]
            except KeyError:
                return jsonify({"error": f"Invalid team: {team_name}"}), 400

            # Get requested tick (default to current tick)
            requested_tick = request.args.get("tick", type=int)
            if requested_tick is None:
                requested_tick = self.knowledge[team].tick

            # Long-polling: wait until knowledge reaches requested tick
            timeout = 30  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_tick = self.knowledge[team].tick
                if current_tick >= requested_tick:
                    # Knowledge is ready
                    return jsonify(
                        {
                            "knowledge": serialize_player_knowledge(
                                self.knowledge[team]
                            ),
                            "base_region": serialize_region(
                                self.state.base_regions[team]
                            ),
                        }
                    )

                time.sleep(0.05)

            # Timeout - return current knowledge anyway
            return jsonify(
                {
                    "knowledge": serialize_player_knowledge(self.knowledge[team]),
                    "base_region": serialize_region(self.state.base_regions[team]),
                }
            )

        @self.app.route("/act/<team_name>/<int:unit_id>", methods=["POST"])
        def set_unit_plan(team_name: str, unit_id: int) -> Any:
            """Set a unit's plan.

            Body: JSON-serialized Plan
            """
            try:
                team = Team[team_name]
            except KeyError:
                return jsonify({"error": f"Invalid team: {team_name}"}), 400

            try:
                plan_data = request.get_json()
                plan = deserialize_plan(plan_data)
            except Exception as e:
                return jsonify({"error": f"Invalid plan data: {e}"}), 400

            try:
                self.state.set_unit_plan(UnitId(unit_id), plan)
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"error": f"Failed to set plan: {e}"}), 400

    def start(self) -> None:
        """Start the server in a background thread."""
        from werkzeug.serving import make_server

        if self.server_thread is not None:
            raise RuntimeError("Server already started")

        # Disable Flask's default logging for cleaner output
        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        # Create server (binds socket immediately)
        server = make_server("0.0.0.0", self.port, self.app, threaded=True)
        ready_event = self.ready_event

        def run_server() -> None:
            # Signal readiness from within the thread, just before serve_forever()
            if ready_event is not None:
                ready_event.set()
            server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        print(f"Server started on port {self.port}")

    def stop(self) -> None:
        """Stop the server (note: Flask doesn't support graceful shutdown easily)."""
        # In a production app, we'd use a proper WSGI server with shutdown support
        # For now, the daemon thread will be killed when the main program exits
        pass
