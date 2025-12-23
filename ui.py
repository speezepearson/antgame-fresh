"""Pygame rendering and UI."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Any
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
    UnitId,
    Order,
    Condition,
    Action,
    Interrupt,
    EnemyInRangeCondition,
    BaseVisibleCondition,
    PositionReachedCondition,
    FoodConfig,
)


# Rendering Constants
TILE_SIZE = 16


@dataclass
class TickControls:
    """UI controls for time navigation."""
    slider: pygame_gui.elements.UIHorizontalSlider
    tick_label: pygame_gui.elements.UILabel
    live_btn: pygame_gui.elements.UIButton


@dataclass
class PlanControls:
    """UI controls for plan editing."""
    text_box: pygame_gui.elements.UITextBox
    last_plan_html: str
    issue_plan_btn: pygame_gui.elements.UIButton
    clear_plan_btn: pygame_gui.elements.UIButton
    selection_label: pygame_gui.elements.UILabel


@dataclass
class PlayerView:
    knowledge: PlayerKnowledge
    tick_controls: TickControls
    plan_controls: PlanControls | None = None
    freeze_frame: Timestamp | None = None
    selected_unit_id: UnitId | None = None
    working_plan: Plan | None = None


@dataclass
class GameContext:
    """Container for all game state and UI elements."""
    state: GameState
    screen: pygame.Surface
    ui_manager: pygame_gui.UIManager
    clock: pygame.time.Clock
    views: dict[Team, PlayerView]
    team_offsets: dict[Team, int]
    red_offset_x: int
    god_offset_x: int
    blue_offset_x: int
    views_offset_y: int
    slider_y: int
    map_pixel_size: int
    tick_interval: int
    last_tick: int


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
            lines.append(
                f"  If {condition_desc}: {'; '.join([action.description for action in interrupt.actions])}"
            )

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
    outline_only: bool = False,  # TODO
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
    surface: pygame.Surface,
    food: dict[Pos, int],
    offset_x: int,
    offset_y: int,
    outline_only: bool = False,
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
                    surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius, 1
                )
            else:
                pygame.draw.circle(
                    surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius
                )


def draw_god_view(
    surface: pygame.Surface,
    state: GameState,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw the god's-eye view showing the complete game state."""
    map_pixel_width = state.grid_width * TILE_SIZE
    map_pixel_height = state.grid_height * TILE_SIZE
    bg_rect = pygame.Rect(offset_x, offset_y, map_pixel_width, map_pixel_height)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    # Draw grid
    draw_grid(surface, offset_x, offset_y, state.grid_width, state.grid_height)

    # Draw base regions
    for pos in state.get_base_region(Team.RED).cells:
        draw_base_cell(surface, pos, Team.RED, offset_x, offset_y)
    for pos in state.get_base_region(Team.BLUE).cells:
        draw_base_cell(surface, pos, Team.BLUE, offset_x, offset_y)

    # Draw food
    draw_food(surface, state.food, offset_x, offset_y)

    # Draw units
    for unit in state.units.values():
        draw_unit_at(
            surface,
            unit.team,
            unit.pos,
            offset_x,
            offset_y,
        )


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
    draw_grid(
        surface,
        offset_x,
        offset_y,
        view.knowledge.grid_width,
        view.knowledge.grid_height,
    )

    if freeze_frame is None:
        for pos, (t, contents_list) in view.knowledge.last_observations.items():
            if freeze_frame is not None and t != view.knowledge.tick:
                continue
            for contents in sorted(
                contents_list,
                key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent)),
            ):
                if isinstance(contents, BasePresent):
                    draw_base_cell(
                        surface,
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(
                        surface,
                        contents.team,
                        pos,
                        offset_x,
                        offset_y,
                        outline_only=pos not in cur_observations,
                    )
                elif isinstance(contents, FoodPresent):
                    draw_food(
                        surface,
                        {pos: contents.count},
                        offset_x,
                        offset_y,
                        outline_only=pos not in cur_observations,
                    )

    else:
        for pos, contents_list in view.knowledge.all_observations.get(
            freeze_frame, {}
        ).items():
            for contents in sorted(
                contents_list,
                key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent)),
            ):
                if isinstance(contents, BasePresent):
                    draw_base_cell(
                        surface,
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(
                        surface,
                        contents.team,
                        pos,
                        offset_x,
                        offset_y,
                        outline_only=False,
                    )
                elif isinstance(contents, FoodPresent):
                    draw_food(
                        surface,
                        {pos: contents.count},
                        offset_x,
                        offset_y,
                        outline_only=False,
                    )

    # Draw predicted positions for units with expected trajectories
    for trajectory in view.knowledge.expected_trajectories.values():
        # Calculate which position in trajectory corresponds to view_t
        trajectory_index = view_t - trajectory.start_tick
        if 0 <= trajectory_index:
            predicted_pos = trajectory.positions[
                min(trajectory_index, len(trajectory.positions) - 1)
            ]
            draw_unit_at(
                surface, team, predicted_pos, offset_x, offset_y, outline_only=True
            )


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
    for unit in state.units.values():
        if unit.team == team and unit.pos == pos:
            if base_region.contains(unit.pos):
                return unit
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


def initialize_game() -> GameContext:
    """Parse arguments and initialize game state, pygame, and UI elements."""
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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    pygame.init()

    # Layout: RED view (with slider) | GOD view | BLUE view (with slider)
    padding = 10
    label_height = 25
    slider_height = 30
    plan_area_height = 240  # Space for plan display and buttons below player maps

    # Create the game state first to get grid dimensions
    food_config = FoodConfig(
        scale=args.food_scale,
        max_prob=args.food_max_prob,
    )
    state = make_game(
        grid_width=args.width, grid_height=args.height, food_config=food_config
    )
    map_pixel_size = state.grid_width * TILE_SIZE

    window_width = map_pixel_size * 3 + padding * 4
    window_height = (
        map_pixel_size + padding * 2 + label_height + slider_height + plan_area_height
    )

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ant RTS")

    # Initialize pygame_gui manager
    ui_manager = pygame_gui.UIManager((window_width, window_height))

    clock = pygame.time.Clock()

    red_offset_x = padding
    god_offset_x = padding * 2 + map_pixel_size
    blue_offset_x = padding * 3 + map_pixel_size * 2
    views_offset_y = padding + label_height
    slider_y = views_offset_y + map_pixel_size + 5
    slider_width = map_pixel_size - 100

    # Map teams to their view offsets
    team_offsets = {
        Team.RED: red_offset_x,
        Team.BLUE: blue_offset_x,
    }

    # Create time sliders for RED team
    red_tick_controls = TickControls(
        slider=pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(red_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        ),
        tick_label=pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        ),
    )

    # Create time sliders for BLUE team
    blue_tick_controls = TickControls(
        slider=pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(blue_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        ),
        tick_label=pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        ),
    )

    tick_interval = 200
    last_tick = pygame.time.get_ticks()

    red_view = PlayerView(
        knowledge=PlayerKnowledge(
            team=Team.RED,
            grid_width=args.width,
            grid_height=args.height,
            tick=state.tick,
        ),
        tick_controls=red_tick_controls,
    )
    blue_view = PlayerView(
        knowledge=PlayerKnowledge(
            team=Team.BLUE,
            grid_width=args.width,
            grid_height=args.height,
            tick=state.tick,
        ),
        tick_controls=blue_tick_controls,
    )
    views = {
        Team.RED: red_view,
        Team.BLUE: blue_view,
    }

    return GameContext(
        state=state,
        screen=screen,
        ui_manager=ui_manager,
        clock=clock,
        views=views,
        team_offsets=team_offsets,
        red_offset_x=red_offset_x,
        god_offset_x=god_offset_x,
        blue_offset_x=blue_offset_x,
        views_offset_y=views_offset_y,
        slider_y=slider_y,
        map_pixel_size=map_pixel_size,
        tick_interval=tick_interval,
        last_tick=last_tick,
    )


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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    pygame.init()

    # Layout: RED view (with slider) | GOD view | BLUE view (with slider)
    padding = 10
    label_height = 25
    slider_height = 30
    plan_area_height = 240  # Space for plan display and buttons below player maps

    # Create the game state first to get grid dimensions
    food_config = FoodConfig(
        scale=args.food_scale,
        max_prob=args.food_max_prob,
    )
    state = make_game(
        grid_width=args.width, grid_height=args.height, food_config=food_config
    )
    map_pixel_size = state.grid_width * TILE_SIZE

    window_width = map_pixel_size * 3 + padding * 4
    window_height = (
        map_pixel_size + padding * 2 + label_height + slider_height + plan_area_height
    )

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ant RTS")

    # Initialize pygame_gui manager
    ui_manager = pygame_gui.UIManager((window_width, window_height))

    clock = pygame.time.Clock()

    red_offset_x = padding
    god_offset_x = padding * 2 + map_pixel_size
    blue_offset_x = padding * 3 + map_pixel_size * 2
    views_offset_y = padding + label_height
    slider_y = views_offset_y + map_pixel_size + 5
    slider_width = map_pixel_size - 100

    # Map teams to their view offsets
    team_offsets = {
        Team.RED: red_offset_x,
        Team.BLUE: blue_offset_x,
    }

    # Create time sliders for RED team
    red_tick_controls = TickControls(
        slider=pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(red_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        ),
        tick_label=pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(red_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        ),
    )

    # Create time sliders for BLUE team
    blue_tick_controls = TickControls(
        slider=pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(blue_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        ),
        tick_label=pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 5, slider_y, 45, 20),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(blue_offset_x + slider_width + 50, slider_y, 50, 20),
            text="LIVE",
            manager=ui_manager,
        ),
    )

    tick_interval = 200
    last_tick = pygame.time.get_ticks()

    red_view = PlayerView(
        knowledge=PlayerKnowledge(
            team=Team.RED,
            grid_width=args.width,
            grid_height=args.height,
            tick=state.tick,
        ),
        tick_controls=red_tick_controls,
    )
    blue_view = PlayerView(
        knowledge=PlayerKnowledge(
            team=Team.BLUE,
            grid_width=args.width,
            grid_height=args.height,
            tick=state.tick,
        ),
        tick_controls=blue_tick_controls,
    )
    views = {
        Team.RED: red_view,
        Team.BLUE: blue_view,
    }

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            # Process pygame_gui events
            if ui_manager.process_events(event):
                continue

            if event.type == pygame.QUIT:
                running = False

            for team in Team:
                view = views[team]

                # Handle pygame_gui button clicks
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    # Handle plan control buttons
                    if event.ui_element == view.tick_controls.live_btn:
                        view.freeze_frame = None
                    if view.plan_controls is not None:
                        if event.ui_element == view.plan_controls.issue_plan_btn:
                            # Issue plan for this team
                            if (
                                view.working_plan is not None
                                and view.working_plan.orders
                                and view.selected_unit_id is not None
                            ):
                                state.set_unit_plan(view.selected_unit_id, view.working_plan)
                                view.selected_unit_id = None
                                view.working_plan = None
                                # Clean up plan controls
                                view.plan_controls.text_box.kill()
                                view.plan_controls.issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                                view.plan_controls.clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                                view.plan_controls.selection_label.hide()  # type: ignore[no-untyped-call]
                                view.plan_controls = None
                            break
                        elif event.ui_element == view.plan_controls.clear_plan_btn:
                            # Clear plan for this team
                            if view.selected_unit_id is not None:
                                view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                            break

                # Handle slider value changes
                elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == view.tick_controls.slider:
                        if state.tick > 0:
                            view.freeze_frame = int(event.value * state.tick)

                # Handle map clicks for unit selection and waypoints
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    # Skip if click is on a pygame_gui element
                    if ui_manager.get_hovering_any_element():
                        continue

                    # Check each team's player view for unit selection or target
                    grid_pos = screen_to_grid(
                        mx,
                        my,
                        team_offsets[team],
                        views_offset_y,
                        state.grid_width,
                        state.grid_height,
                    )
                    # Only allow interaction when viewing live (not a freeze frame)
                    if grid_pos is not None and view.freeze_frame is None:
                        if view.selected_unit_id is not None:
                            # Append Move order to working plan
                            if view.working_plan is None:
                                view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )
                            view.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base(state, grid_pos, team)
                            if unit is not None:
                                view.selected_unit_id = unit.id
                                # Initialize working plan when selecting a unit
                                view.working_plan = Plan(
                                    interrupts=make_default_interrupts()
                                )

        # Game tick
        if current_time - last_tick >= tick_interval:
            tick_game(state)
            for view in views.values():
                view.knowledge.tick_knowledge(state)
            last_tick = current_time

        # Render
        screen.fill((0, 0, 0))

        # Draw game views (these are still custom pygame rendering)
        draw_player_view(
            screen, views[Team.RED], Team.RED, red_offset_x, views_offset_y
        )
        draw_god_view(screen, state, god_offset_x, views_offset_y)
        draw_player_view(
            screen, views[Team.BLUE], Team.BLUE, blue_offset_x, views_offset_y
        )

        # Update slider positions and labels to reflect current state
        if state.tick > 0:
            red_view = views[Team.RED]
            red_view_tick = red_view.freeze_frame
            if red_view_tick is None:
                red_view.tick_controls.slider.set_current_value(1.0)
                red_view.tick_controls.tick_label.set_text(f"t={state.tick}")
            else:
                red_view.tick_controls.slider.set_current_value(red_view_tick / state.tick)
                red_view.tick_controls.tick_label.set_text(f"t={red_view_tick}")

            blue_view = views[Team.BLUE]
            blue_view_tick = blue_view.freeze_frame
            if blue_view_tick is None:
                blue_view.tick_controls.slider.set_current_value(1.0)
                blue_view.tick_controls.tick_label.set_text(f"t={state.tick}")
            else:
                blue_view.tick_controls.slider.set_current_value(blue_view_tick / state.tick)
                blue_view.tick_controls.tick_label.set_text(f"t={blue_view_tick}")

        # Plan area layout
        plan_y = slider_y + 30
        plan_box_height = 180
        btn_y = plan_y + plan_box_height + 5
        selection_label_y = slider_y

        # Handle plan controls for each team
        for team in Team:
            view = views[team]
            plan_offset_x = team_offsets[team]

            if view.selected_unit_id is not None:
                # Get the selected unit
                selected_unit = state.units.get(view.selected_unit_id)
                if selected_unit is None:
                    # Unit no longer exists, clear selection
                    view.selected_unit_id = None
                    view.working_plan = None
                    if view.plan_controls is not None:
                        view.plan_controls.text_box.kill()
                        view.plan_controls.issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                        view.plan_controls.clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                        view.plan_controls.selection_label.hide()  # type: ignore[no-untyped-call]
                        view.plan_controls = None
                    continue

                team_name = team.value

                # Create plan controls if they don't exist
                if view.plan_controls is None:
                    selection_label = pygame_gui.elements.UILabel(
                        relative_rect=pygame.Rect(plan_offset_x, selection_label_y, map_pixel_size, 20),
                        text="",
                        manager=ui_manager,
                    )
                    text_box = pygame_gui.elements.UITextBox(
                        html_text="",
                        relative_rect=pygame.Rect(
                            plan_offset_x, plan_y, map_pixel_size, plan_box_height
                        ),
                        manager=ui_manager,
                        wrap_to_height=False,
                    )
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
                    view.plan_controls = PlanControls(
                        text_box=text_box,
                        last_plan_html="",
                        issue_plan_btn=issue_plan_btn,
                        clear_plan_btn=clear_plan_btn,
                        selection_label=selection_label,
                    )

                # Update selection label
                view.plan_controls.selection_label.set_text(
                    f"Selected: {team_name} unit - click {team_name}'s map to add waypoints"
                )
                view.plan_controls.selection_label.show()  # type: ignore[no-untyped-call]

                # Display the working plan in a scrollable text box
                if view.working_plan is not None:
                    plan_lines = format_plan(view.working_plan, selected_unit)
                    plan_html = "<br>".join(plan_lines)

                    # Only update if content changed
                    if plan_html != view.plan_controls.last_plan_html:
                        view.plan_controls.text_box.set_text(plan_html)
                        view.plan_controls.text_box.rebuild()
                        view.plan_controls.last_plan_html = plan_html

                # Enable/disable Issue Plan button based on whether there are orders
                if view.working_plan and view.working_plan.orders:
                    view.plan_controls.issue_plan_btn.enable()  # type: ignore[no-untyped-call]
                else:
                    view.plan_controls.issue_plan_btn.disable()  # type: ignore[no-untyped-call]

            else:
                # No selected unit for this team, clean up plan controls if they exist
                if view.plan_controls is not None:
                    view.plan_controls.text_box.kill()
                    view.plan_controls.issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                    view.plan_controls.clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                    view.plan_controls.selection_label.hide()  # type: ignore[no-untyped-call]
                    view.plan_controls = None

        # Update UI manager
        ui_manager.update(time_delta)

        # Draw UI manager
        ui_manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
