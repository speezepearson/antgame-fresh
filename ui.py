"""Pygame rendering and UI."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
import threading
from typing import Any, Literal
import argparse
import numpy
import pygame
import pygame_gui

from core import Pos, Region, Timestamp
from knowledge import PlayerKnowledge
from mechanics import (
    Empty,
    FoodInRangeCondition,
    FoodPresent,
    GameState,
    MoveHomeAction,
    MoveThereAction,
    Team,
    BasePresent,
    UnitPresent,
    GRID_SIZE,
    Move,
    Plan,
    make_game,
    tick_game,
    Unit,
    Order,
    Condition,
    Action,
    Interrupt,
    EnemyInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodConfig,
)
from client import GameClient, LocalGameClient, RemoteGameClient
from server import create_server

GameMode = Literal["local", "server", "client"]


# Rendering Constants
TILE_SIZE = 16

@dataclass
class PlayerView:
    knowledge: PlayerKnowledge
    freeze_frame: Timestamp | None = None
    selected_unit: Unit | None = None
    working_plan: Plan | None = None


def format_plan(plan: Plan, unit: Unit) -> list[str]:
    """Format a plan as a list of strings for display."""
    lines = []

    # Show current and remaining orders
    if plan.orders:
        lines.append("Orders:")
        for i, order in enumerate(plan.orders):
            prefix = "> " if i == 0 else "  "
            lines.append(f"{prefix}{order.description}")
    else:
        lines.append("Orders: (none)")

    # Show interrupts
    if plan.interrupts:
        lines.append("Interrupts:")
        for interrupt in plan.interrupts:
            condition_desc = interrupt.condition.description
            lines.append(f"  If {condition_desc}: {'; '.join([action.description for action in interrupt.actions])}")

    return lines


def draw_grid(
    surface: pygame.Surface,
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
) -> None:
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    for x in range(grid_width + 1):
        pygame.draw.line(
            surface,
            (50, 50, 50),
            (offset_x + x * TILE_SIZE, offset_y),
            (offset_x + x * TILE_SIZE, offset_y + map_pixel_height),
        )
    for y in range(grid_height + 1):
        pygame.draw.line(
            surface,
            (50, 50, 50),
            (offset_x, offset_y + y * TILE_SIZE),
            (offset_x + map_pixel_width, offset_y + y * TILE_SIZE),
        )


def draw_base_cell(
    surface: pygame.Surface,
    pos: Pos,
    team: Team,
    offset_x: int,
    offset_y: int,
    outline_only: bool = False, # TODO
) -> None:
    """Draw a base region with a faint background tint."""
    # Faint background color for base cells
    tint_color = (80, 40, 40) if team == Team.RED else (40, 40, 200)
    rect = pygame.Rect(
        offset_x + pos.x * TILE_SIZE,
        offset_y + pos.y * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
    )
    pygame.draw.rect(surface, tint_color, rect)


def draw_unit_at(
    surface: pygame.Surface,
    team: Team,
    pos: Pos,
    offset_x: int,
    offset_y: int,
    selected: bool = False,
    outline_only: bool = False,
) -> None:
    color = (255, 100, 100) if team == Team.RED else (100, 100, 255)
    center_x = offset_x + pos.x * TILE_SIZE + TILE_SIZE // 2
    center_y = offset_y + pos.y * TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 3

    if outline_only:
        pygame.draw.circle(surface, color, (center_x, center_y), radius, 1)
    else:
        pygame.draw.circle(surface, color, (center_x, center_y), radius)
    if selected:
        pygame.draw.circle(surface, (255, 255, 0), (center_x, center_y), radius + 3, 2)


def draw_food(
    surface: pygame.Surface, food: dict[Pos, int], offset_x: int, offset_y: int, outline_only: bool = False
) -> None:
    """Draw food as small green dots, with multiple items positioned non-overlapping."""
    radius = 3

    for pos, count in food.items():
        tile_x = offset_x + pos.x * TILE_SIZE
        tile_y = offset_y + pos.y * TILE_SIZE

        if count == 1:
            # Single item: center of tile
            positions = [(TILE_SIZE // 2, TILE_SIZE // 2)]
        elif count == 2:
            # Two items: left and right
            positions = [
                (TILE_SIZE // 3, TILE_SIZE // 2),
                (2 * TILE_SIZE // 3, TILE_SIZE // 2),
            ]
        elif count == 3:
            # Three items: triangle pattern
            positions = [
                (TILE_SIZE // 2, TILE_SIZE // 3),  # top center
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),  # bottom left
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),  # bottom right
            ]
        elif count == 4:
            # Four items: corners
            positions = [
                (TILE_SIZE // 3, TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, TILE_SIZE // 3),
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),
            ]
        else:
            # Five or more: 2x2 grid plus center
            positions = [
                (TILE_SIZE // 3, TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, TILE_SIZE // 3),
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (TILE_SIZE // 2, TILE_SIZE // 2),
            ][: min(count, 5)]

        for dx, dy in positions:
            if outline_only:
                pygame.draw.circle(
                    surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius, 1)
            else:
                pygame.draw.circle(
                    surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius
                )

def god_knowledge(game: GameState) -> PlayerKnowledge:
    observations = game.observe_from_position(Pos(0, 0), (game.grid_width+game.grid_height))
    return PlayerKnowledge(
        team=Team.RED,  # TODO: asymmetric
        grid_width=game.grid_width,
        grid_height=game.grid_height,
        tick=game.tick,
        all_observations={game.tick: observations},
        last_observations={pos: (game.tick, contents_list) for pos, contents_list in observations.items()},
    )

def draw_god_view(
    surface: pygame.Surface,
    state: GameState,
    offset_x: int,
    offset_y: int,
) -> None:
    map_pixel_width = state.grid_width * TILE_SIZE
    map_pixel_height = state.grid_height * TILE_SIZE
    bg_rect = pygame.Rect(offset_x, offset_y, map_pixel_width, map_pixel_height)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    knowledge = god_knowledge(state)
    # breakpoint()
    draw_player_view(surface, PlayerView(knowledge), Team.RED, offset_x, offset_y)

    # draw_grid(surface, offset_x, offset_y, state.grid_width, state.grid_height)
    # draw_base_cell(surface, state.get_base_region(Team.RED), Team.RED, offset_x, offset_y)
    # draw_base_cell(surface, state.get_base_region(Team.BLUE), Team.BLUE, offset_x, offset_y)

    # # Draw food
    # draw_food(surface, state.food, offset_x, offset_y)

    # for unit in state.units:
    #     draw_unit_at(
    #         surface,
    #         unit.team,
    #         unit.pos,
    #         offset_x,
    #         offset_y,
    #     )


def draw_player_view(
    surface: pygame.Surface,
    view: PlayerView,
    team: Team,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw a player's view of the map at their selected tick."""
    freeze_frame = view.freeze_frame
    # Use current tick when live (freeze_frame is None), otherwise use freeze_frame
    view_t = view.knowledge.tick if freeze_frame is None else freeze_frame
    logbook = view.knowledge.all_observations

    map_pixel_width = view.knowledge.grid_width * TILE_SIZE
    map_pixel_height = view.knowledge.grid_height * TILE_SIZE
    bg_rect = pygame.Rect(offset_x, offset_y, map_pixel_width, map_pixel_height)
    pygame.draw.rect(surface, (20, 20, 20), bg_rect)

    # Get observations for this timestamp
    cur_observations = logbook.get(view_t, {})

    # Draw visible tiles as slightly lighter
    for pos in cur_observations.keys():
        rect = pygame.Rect(
            offset_x + pos.x * TILE_SIZE,
            offset_y + pos.y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(surface, (40, 40, 40), rect)

    # Redraw grid on top
    draw_grid(surface, offset_x, offset_y, view.knowledge.grid_width, view.knowledge.grid_height)

    if freeze_frame is None:
        for pos, (t, contents_list) in view.knowledge.last_observations.items():
            if (freeze_frame is not None and t != view.knowledge.tick): continue
            for contents in sorted(contents_list, key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent))):
                if isinstance(contents, BasePresent):
                    draw_base_cell(
                        surface,
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(surface, contents.team, pos, offset_x, offset_y, outline_only=pos not in cur_observations)
                elif isinstance(contents, FoodPresent):
                    draw_food(surface, {pos: contents.count}, offset_x, offset_y, outline_only=pos not in cur_observations)

    else:
        for pos, contents_list in view.knowledge.all_observations.get(freeze_frame, {}).items():
            for contents in sorted(contents_list, key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent))):
                if isinstance(contents, BasePresent):
                    draw_base_cell(
                        surface,
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(surface, contents.team, pos, offset_x, offset_y, outline_only=False)
                elif isinstance(contents, FoodPresent):
                    draw_food(surface, {pos: contents.count}, offset_x, offset_y, outline_only=False)


    # Draw predicted positions for units with expected trajectories
    for trajectory in view.knowledge.expected_trajectories.values():
        # Calculate which position in trajectory corresponds to view_t
        trajectory_index = view_t - trajectory.start_tick
        if 0 <= trajectory_index:
            predicted_pos = trajectory.positions[min(trajectory_index, len(trajectory.positions) - 1)]
            draw_unit_at(surface, team, predicted_pos, offset_x, offset_y, outline_only=True)


def screen_to_grid(
    mouse_x: int,
    mouse_y: int,
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
) -> Pos | None:
    rel_x = mouse_x - offset_x
    rel_y = mouse_y - offset_y

    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    if 0 <= rel_x < map_pixel_width and 0 <= rel_y < map_pixel_height:
        return Pos(rel_x // TILE_SIZE, rel_y // TILE_SIZE)
    return None


def find_unit_at_base(state: GameState, pos: Pos, team: Team) -> Unit | None:
    """Find a unit of the given team at pos, only if inside their base region."""
    base_region = state.get_base_region(team)
    for unit in state.units:
        if unit.team == team and unit.pos == pos:
            if base_region.contains(unit.pos):
                return unit
    return None


def find_unit_at_base_from_knowledge(
    knowledge: PlayerKnowledge, pos: Pos, team: Team, state: GameState | None = None
) -> Unit | None:
    """Find a unit from knowledge at pos, only if inside their base region.

    In client mode (state=None), we check if pos is in a base from observations.
    In local/server mode, we use the actual GameState to find the unit.
    """
    if state is not None:
        # Local/server mode: use the actual game state
        return find_unit_at_base(state, pos, team)

    # Client mode: find unit from knowledge observations
    # Check if there's a base at this position
    last_obs = knowledge.last_observations.get(pos)
    if last_obs is None:
        return None

    _, contents_list = last_obs
    has_base = any(isinstance(c, BasePresent) and c.team == team for c in contents_list)
    if not has_base:
        return None

    # Find unit at this position
    for contents in contents_list:
        if isinstance(contents, UnitPresent) and contents.team == team:
            # Create a minimal Unit object for selection purposes
            # Note: we don't have full unit data in client mode, but we have the ID
            return Unit(
                id=contents.unit_id,
                team=contents.team,
                pos=pos,
                original_pos=pos,  # We don't know the original pos, so use current
            )

    return None


def make_default_interrupts() -> list[Interrupt[Any]]:
    return [
        Interrupt(
            condition=EnemyInRangeCondition(distance=2),
            actions=[MoveHomeAction() if random.random() < 0.5 else MoveThereAction()],
        ),
        Interrupt(
            condition=FoodInRangeCondition(distance=2),
            actions=[MoveThereAction(), MoveHomeAction()],
        ),
    ]


def main() -> None:
    # Parse command-line arguments
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
        help="Game mode: local (single player), server (host multiplayer), or client (join multiplayer)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:5000",
        help="Server URL for client mode (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--team",
        type=str,
        choices=["RED", "BLUE"],
        default=None,
        help="Team to play as in client mode (required for client mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for server mode (default: 5000)",
    )
    args = parser.parse_args()

    # Validate client mode requirements
    if args.mode == "client" and args.team is None:
        parser.error("--team is required in client mode")

    mode: GameMode = args.mode

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    pygame.init()

    # Initialize game state and client based on mode
    state: GameState | None = None
    client: GameClient
    client_team: Team | None = None
    flask_server = None

    if mode in ["local", "server"]:
        # Create game state locally
        food_config = FoodConfig(
            scale=args.food_scale,
            max_prob=args.food_max_prob,
        )
        state = make_game(
            grid_width=args.width, grid_height=args.height, food_config=food_config
        )

        # Initialize knowledge for both teams
        red_view = PlayerView(knowledge=PlayerKnowledge(team=Team.RED, grid_width=args.width, grid_height=args.height, tick=state.tick))
        blue_view = PlayerView(knowledge=PlayerKnowledge(team=Team.BLUE, grid_width=args.width, grid_height=args.height, tick=state.tick))
        views = {
            Team.RED: red_view,
            Team.BLUE: blue_view,
        }

        # Create local client
        client = LocalGameClient(state, {Team.RED: red_view.knowledge, Team.BLUE: blue_view.knowledge})

        # Start Flask server in server mode
        if mode == "server":
            flask_app = create_server(state, {Team.RED: red_view.knowledge, Team.BLUE: blue_view.knowledge}, port=args.port)
            flask_thread = threading.Thread(
                target=lambda: flask_app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False),
                daemon=True,
            )
            flask_thread.start()
            print(f"Server started on port {args.port}")

    else:  # client mode
        client_team = Team(args.team)
        client = RemoteGameClient(args.server_url, client_team)

        # Fetch initial knowledge from server
        initial_knowledge = client.get_knowledge(client_team, 0)
        views = {
            client_team: PlayerView(knowledge=initial_knowledge),
        }

        # We don't have access to GameState in client mode
        state = None

    # Determine layout based on mode
    if mode == "client":
        # Client mode: only show the player's team view
        assert client_team is not None, "client_team must be set in client mode"
        num_views = 1
        map_pixel_size = views[client_team].knowledge.grid_width * TILE_SIZE
    else:
        # Local/server mode: show all three views
        assert state is not None, "state must be set in local/server mode"
        num_views = 3
        map_pixel_size = state.grid_width * TILE_SIZE

    # Layout
    padding = 10
    label_height = 25
    slider_height = 30
    plan_area_height = 240  # Space for plan display and buttons below player maps

    window_width = map_pixel_size * num_views + padding * (num_views + 1)
    window_height = (
        map_pixel_size + padding * 2 + label_height + slider_height + plan_area_height
    )

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Ant RTS - {mode.upper()} mode")

    # Initialize pygame_gui manager
    ui_manager = pygame_gui.UIManager((window_width, window_height))

    clock = pygame.time.Clock()

    # Calculate view offsets based on mode
    views_offset_y = padding + label_height
    slider_y = views_offset_y + map_pixel_size + 5
    slider_width = map_pixel_size - 100

    if mode == "client":
        # Client mode: single view centered
        assert client_team is not None
        view_offset_x = padding
        team_name = client_team.value
        team_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(view_offset_x, padding, 100, label_height),
            text=f"{team_name}'S MAP",
            manager=ui_manager,
        )
        team_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(view_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        )
        team_tick_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(view_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        )
        team_live_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(view_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        )
        # Store references for event handling
        sliders_and_buttons = {
            client_team: (team_slider, team_tick_label, team_live_btn)
        }
        red_offset_x = view_offset_x
        god_offset_x = None
        blue_offset_x = None
    else:
        # Local/server mode: three views
        red_offset_x = padding
        god_offset_x = padding * 2 + map_pixel_size
        blue_offset_x = padding * 3 + map_pixel_size * 2

        # Create pygame_gui labels
        red_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(red_offset_x, padding, 100, label_height),
            text="RED'S MAP",
            manager=ui_manager,
        )
        god_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(god_offset_x, padding, 120, label_height),
            text="GOD'S EYE VIEW",
            manager=ui_manager,
        )
        blue_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(blue_offset_x, padding, 100, label_height),
            text="BLUE'S MAP",
            manager=ui_manager,
        )

        # Create time sliders for each team in local/server mode
        red_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(red_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        )
        red_tick_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        )
        red_live_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        )

        blue_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(blue_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        )
        blue_tick_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        )
        blue_live_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        )

        # Store references for event handling
        sliders_and_buttons = {
            Team.RED: (red_slider, red_tick_label, red_live_btn),
            Team.BLUE: (blue_slider, blue_tick_label, blue_live_btn),
        }

    # Selection indicator label (hidden when no selection)
    selection_label_x = god_offset_x if god_offset_x is not None else view_offset_x
    selection_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(selection_label_x, slider_y, map_pixel_size, 20),
        text="",
        manager=ui_manager,
    )
    selection_label.hide()  # type: ignore[no-untyped-call]

    # Plan text box and buttons (created dynamically per-team)
    plan_text_box: pygame_gui.elements.UITextBox | None = None
    last_plan_html: str | None = None
    issue_plan_btn: pygame_gui.elements.UIButton | None = None
    clear_plan_btn: pygame_gui.elements.UIButton | None = None
    plan_buttons_team: Team | None = None  # Track which team the buttons are for

    tick_interval = 200
    last_tick = pygame.time.get_ticks()
    last_knowledge_poll = pygame.time.get_ticks()

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            # Process pygame_gui events
            ui_manager.process_events(event)

            if event.type == pygame.QUIT:
                running = False

            # Handle pygame_gui button clicks
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                # Handle live buttons for all teams
                for team, (slider, tick_label, live_btn) in sliders_and_buttons.items():
                    if event.ui_element == live_btn:
                        views[team].freeze_frame = None
                        break

                if event.ui_element == issue_plan_btn:
                    # Issue plan for the active team using client interface
                    for team in views.keys():
                        view = views[team]
                        if (
                            view.working_plan is not None
                            and view.working_plan.orders
                            and view.selected_unit is not None
                        ):
                            # Use client interface to set the plan
                            success = client.set_plan(team, view.selected_unit.id, view.working_plan)
                            if success:
                                view.selected_unit = None
                                view.working_plan = None
                            else:
                                print(f"Failed to set plan for unit {view.selected_unit.id}")
                            break
                elif event.ui_element == clear_plan_btn:
                    # Clear plan for the active team
                    for team in views.keys():
                        view = views[team]
                        if view.selected_unit is not None:
                            view.working_plan = Plan(interrupts=make_default_interrupts())
                            break

            # Handle slider value changes
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                for team, (slider, tick_label, live_btn) in sliders_and_buttons.items():
                    if event.ui_element == slider:
                        current_tick = views[team].knowledge.tick
                        if current_tick > 0:
                            views[team].freeze_frame = int(event.value * current_tick)
                        break

            # Handle map clicks for unit selection and waypoints
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Skip if click is on a pygame_gui element
                if ui_manager.get_hovering_any_element():
                    continue

                # Handle clicks for each visible team view
                if mode == "client":
                    # Client mode: single team view
                    assert client_team is not None
                    team = client_team
                    view = views[team]
                    grid_pos = screen_to_grid(
                        mx,
                        my,
                        view_offset_x,
                        views_offset_y,
                        view.knowledge.grid_width,
                        view.knowledge.grid_height,
                    )
                    if grid_pos is not None and view.freeze_frame is None:
                        if view.selected_unit is not None:
                            # Append Move order to working plan
                            if view.working_plan is None:
                                view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                            view.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base_from_knowledge(
                                view.knowledge, grid_pos, team, state
                            )
                            if unit is not None:
                                view.selected_unit = unit
                                view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                else:
                    # Local/server mode: check both RED and BLUE views
                    # Check RED player view for unit selection or target
                    grid_pos = screen_to_grid(
                        mx,
                        my,
                        red_offset_x,
                        views_offset_y,
                        state.grid_width,  # type: ignore
                        state.grid_height,  # type: ignore
                    )
                    red_view = views[Team.RED]
                    # Only allow interaction when viewing live (not a freeze frame)
                    if grid_pos is not None and red_view.freeze_frame is None:
                        if red_view.selected_unit is not None:
                            # Append Move order to working plan
                            if red_view.working_plan is None:
                                red_view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                            red_view.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base_from_knowledge(
                                red_view.knowledge, grid_pos, Team.RED, state
                            )
                            if unit is not None:
                                red_view.selected_unit = unit
                                # Initialize working plan when selecting a unit
                                red_view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )

                    # Check BLUE player view for unit selection or target
                    grid_pos = screen_to_grid(
                        mx,
                        my,
                        blue_offset_x,  # type: ignore
                        views_offset_y,
                        state.grid_width,  # type: ignore
                        state.grid_height,  # type: ignore
                    )
                    blue_view = views[Team.BLUE]
                    # Only allow interaction when viewing live (not a freeze frame)
                    if grid_pos is not None and blue_view.freeze_frame is None:
                        if blue_view.selected_unit is not None:
                            # Append Move order to working plan
                            if blue_view.working_plan is None:
                                blue_view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                            blue_view.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base_from_knowledge(
                                blue_view.knowledge, grid_pos, Team.BLUE, state
                            )
                            if unit is not None:
                                blue_view.selected_unit = unit
                                # Initialize working plan when selecting a unit
                                blue_view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )

        # Game tick or knowledge polling
        if mode in ["local", "server"]:
            # Local/server mode: tick the game
            assert state is not None
            if current_time - last_tick >= tick_interval:
                tick_game(state)
                for view in views.values():
                    view.knowledge.tick_knowledge(state)
                last_tick = current_time
        else:
            # Client mode: poll for knowledge updates
            assert client_team is not None
            if current_time - last_knowledge_poll >= 100:  # Poll every 100ms
                try:
                    team = client_team
                    current_tick = views[team].knowledge.tick
                    new_knowledge = client.get_knowledge(team, current_tick + 1)
                    views[team].knowledge = new_knowledge
                except Exception as e:
                    # Ignore polling errors (server might not have new data yet)
                    pass
                last_knowledge_poll = current_time

        # Render
        screen.fill((0, 0, 0))

        # Draw game views based on mode
        if mode == "client":
            # Client mode: only show the player's team view
            assert client_team is not None
            team = client_team
            draw_player_view(screen, views[team], team, view_offset_x, views_offset_y)
        else:
            # Local/server mode: show all three views
            assert state is not None
            assert god_offset_x is not None
            assert blue_offset_x is not None
            draw_player_view(screen, views[Team.RED], Team.RED, red_offset_x, views_offset_y)
            draw_god_view(screen, state, god_offset_x, views_offset_y)
            draw_player_view(screen, views[Team.BLUE], Team.BLUE, blue_offset_x, views_offset_y)

        # Update slider positions and labels to reflect current state
        for team, (slider, tick_label, live_btn) in sliders_and_buttons.items():
            view = views[team]
            current_tick = view.knowledge.tick
            if current_tick > 0:
                view_tick = view.freeze_frame
                if view_tick is None:
                    slider.set_current_value(1.0)
                    tick_label.set_text(f"t={current_tick}")
                else:
                    slider.set_current_value(view_tick / current_tick)
                    tick_label.set_text(f"t={view_tick}")

        # Plan area layout
        plan_y = slider_y + 30
        plan_box_height = 180
        btn_y = plan_y + plan_box_height + 5

        # Find which team (if any) has a selected unit
        active_team: Team | None = None
        for team in views.keys():
            if views[team].selected_unit is not None:
                active_team = team
                break

        if active_team is not None:
            view = views[active_team]
            selected_unit = view.selected_unit
            assert selected_unit is not None  # for type checker
            team_name = active_team.value
            if mode == "client":
                plan_offset_x = view_offset_x
            else:
                plan_offset_x = (
                    red_offset_x if active_team == Team.RED else blue_offset_x  # type: ignore
                )

            # Update selection label
            selection_label.set_text(
                f"Selected: {team_name} unit - click {team_name}'s map to add waypoints"
            )
            selection_label.show()  # type: ignore[no-untyped-call]

            # Display the working plan in a scrollable text box
            if view.working_plan is not None:
                plan_lines = format_plan(view.working_plan, selected_unit)
                plan_html = "<br>".join(plan_lines)

                # Create or update the plan text box
                text_box_rect = pygame.Rect(
                    plan_offset_x, plan_y, map_pixel_size, plan_box_height
                )

                # Only recreate or update if position changed or content changed
                if (
                    plan_text_box is None
                    or plan_text_box.relative_rect != text_box_rect
                ):
                    # Position changed, need to recreate
                    if plan_text_box is not None:
                        plan_text_box.kill()
                    plan_text_box = pygame_gui.elements.UITextBox(
                        html_text=plan_html,
                        relative_rect=text_box_rect,
                        manager=ui_manager,
                        wrap_to_height=False,
                    )
                    last_plan_html = plan_html
                elif plan_html != last_plan_html:
                    # Content changed, update text
                    plan_text_box.set_text(plan_html)
                    plan_text_box.rebuild()
                    last_plan_html = plan_html
                # else: content unchanged, don't touch the text box (preserves scroll)
            elif plan_text_box is not None:
                # No plan, remove the text box
                plan_text_box.kill()
                plan_text_box = None
                last_plan_html = None

            # Create or reposition Issue Plan and Clear buttons
            if plan_buttons_team != active_team:
                # Need to recreate buttons for the new team
                if issue_plan_btn is not None:
                    issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                if clear_plan_btn is not None:
                    clear_plan_btn.kill()  # type: ignore[no-untyped-call]

                issue_plan_btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(plan_offset_x, btn_y, 80, 20),
                    text="Issue Plan",
                    manager=ui_manager,
                )
                clear_plan_btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(plan_offset_x + 90, btn_y, 60, 20),
                    text="Clear",
                    manager=ui_manager,
                )
                plan_buttons_team = active_team

            # Enable/disable Issue Plan button based on whether there are orders
            if issue_plan_btn is not None:
                if view.working_plan and view.working_plan.orders:
                    issue_plan_btn.enable()  # type: ignore[no-untyped-call]
                else:
                    issue_plan_btn.disable()  # type: ignore[no-untyped-call]

        else:
            # No selected unit on either team
            selection_label.hide()  # type: ignore[no-untyped-call]

            if plan_text_box is not None:
                plan_text_box.kill()
                plan_text_box = None
                last_plan_html = None

            if issue_plan_btn is not None:
                issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                issue_plan_btn = None
            if clear_plan_btn is not None:
                clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                clear_plan_btn = None
            plan_buttons_team = None

        # Update UI manager
        ui_manager.update(time_delta)

        # Draw UI manager
        ui_manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
