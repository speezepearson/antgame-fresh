"""Pygame rendering and UI."""

from __future__ import annotations
import random
from typing import Any
import argparse
import pygame
import pygame_gui

from core import Pos, Region
from mechanics import (
    FoodInRangeCondition,
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


# Rendering Constants
TILE_SIZE = 16


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
            lines.append(f"  If {condition_desc}: <action>")

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


def draw_base(
    surface: pygame.Surface, region: Region, team: Team, offset_x: int, offset_y: int
) -> None:
    """Draw a base region with a faint background tint."""
    # Faint background color for base cells
    tint_color = (80, 40, 40) if team == Team.RED else (40, 40, 80)
    for pos in region.cells:
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
) -> None:
    color = (255, 100, 100) if team == Team.RED else (100, 100, 255)
    center_x = offset_x + pos.x * TILE_SIZE + TILE_SIZE // 2
    center_y = offset_y + pos.y * TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 3

    pygame.draw.circle(surface, color, (center_x, center_y), radius)
    if selected:
        pygame.draw.circle(surface, (255, 255, 0), (center_x, center_y), radius + 3, 2)


def draw_food(
    surface: pygame.Surface, food: dict[Pos, int], offset_x: int, offset_y: int
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
            pygame.draw.circle(
                surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius
            )

        # If more than 5, indicate with a brighter center dot
        if count > 5:
            pygame.draw.circle(
                surface,
                (150, 255, 150),
                (tile_x + TILE_SIZE // 2, tile_y + TILE_SIZE // 2),
                radius,
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

    draw_grid(surface, offset_x, offset_y, state.grid_width, state.grid_height)
    draw_base(surface, state.get_base_region(Team.RED), Team.RED, offset_x, offset_y)
    draw_base(surface, state.get_base_region(Team.BLUE), Team.BLUE, offset_x, offset_y)

    # Draw food
    draw_food(surface, state.food, offset_x, offset_y)

    for unit in state.units:
        draw_unit_at(
            surface,
            unit.team,
            unit.pos,
            offset_x,
            offset_y,
            selected=(unit is state.views[unit.team].selected_unit),
        )


def draw_player_view(
    surface: pygame.Surface,
    state: GameState,
    team: Team,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw a player's view of the map at their selected tick."""
    freeze_frame = state.views[team].freeze_frame
    # Use current tick when live (freeze_frame is None), otherwise use freeze_frame
    view_t = state.tick if freeze_frame is None else freeze_frame
    logbook = state.views[team].logbook

    map_pixel_width = state.grid_width * TILE_SIZE
    map_pixel_height = state.grid_height * TILE_SIZE
    bg_rect = pygame.Rect(offset_x, offset_y, map_pixel_width, map_pixel_height)
    pygame.draw.rect(surface, (20, 20, 20), bg_rect)

    draw_grid(surface, offset_x, offset_y, state.grid_width, state.grid_height)

    # Get observations for this timestamp
    if view_t in logbook:
        observations = logbook[view_t]

        # Draw visible tiles as slightly lighter
        for pos in observations.keys():
            rect = pygame.Rect(
                offset_x + pos.x * TILE_SIZE,
                offset_y + pos.y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )
            pygame.draw.rect(surface, (40, 40, 40), rect)

        # Redraw grid on top
        draw_grid(surface, offset_x, offset_y, state.grid_width, state.grid_height)

        # Draw observed contents (bases first, then units on top)
        drawn_bases = set()
        # First pass: draw bases
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, BasePresent):
                    if contents.team not in drawn_bases:
                        draw_base(
                            surface,
                            state.get_base_region(contents.team),
                            contents.team,
                            offset_x,
                            offset_y,
                        )
                        drawn_bases.add(contents.team)
        # Second pass: draw food in visible areas
        visible_food = {
            pos: count for pos, count in state.food.items() if pos in observations
        }
        draw_food(surface, visible_food, offset_x, offset_y)
        # Third pass: draw units on top
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, UnitPresent):
                    draw_unit_at(surface, contents.team, pos, offset_x, offset_y)


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
        "--food-seed",
        type=int,
        default=None,
        help="Random seed for food generation (default: random)",
    )
    args = parser.parse_args()

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
        seed=args.food_seed,
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

    # Create time sliders for each team
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

    # Selection indicator label (hidden when no selection)
    selection_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(god_offset_x, slider_y, map_pixel_size, 20),
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
                if event.ui_element == red_live_btn:
                    state.views[Team.RED].freeze_frame = None
                elif event.ui_element == blue_live_btn:
                    state.views[Team.BLUE].freeze_frame = None
                elif event.ui_element == issue_plan_btn:
                    # Issue plan for the active team
                    for team in [Team.RED, Team.BLUE]:
                        view = state.views[team]
                        if (
                            view.working_plan is not None
                            and view.working_plan.orders
                            and view.selected_unit is not None
                        ):
                            view.selected_unit.plan = view.working_plan
                            view.selected_unit = None
                            view.working_plan = None
                            break
                elif event.ui_element == clear_plan_btn:
                    # Clear plan for the active team
                    for team in [Team.RED, Team.BLUE]:
                        view = state.views[team]
                        if view.selected_unit is not None:
                            view.working_plan = Plan(interrupts=make_default_interrupts())
                            break

            # Handle slider value changes
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == red_slider:
                    if state.tick > 0:
                        state.views[Team.RED].freeze_frame = int(event.value * state.tick)
                elif event.ui_element == blue_slider:
                    if state.tick > 0:
                        state.views[Team.BLUE].freeze_frame = int(event.value * state.tick)

            # Handle map clicks for unit selection and waypoints
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Skip if click is on a pygame_gui element
                if ui_manager.get_hovering_any_element():
                    continue

                # Check RED player view for unit selection or target
                grid_pos = screen_to_grid(
                    mx,
                    my,
                    red_offset_x,
                    views_offset_y,
                    state.grid_width,
                    state.grid_height,
                )
                red_view = state.views[Team.RED]
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
                        unit = find_unit_at_base(state, grid_pos, Team.RED)
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
                    blue_offset_x,
                    views_offset_y,
                    state.grid_width,
                    state.grid_height,
                )
                blue_view = state.views[Team.BLUE]
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
                        unit = find_unit_at_base(state, grid_pos, Team.BLUE)
                        if unit is not None:
                            blue_view.selected_unit = unit
                            # Initialize working plan when selecting a unit
                            blue_view.working_plan = Plan(
                                interrupts=make_default_interrupts()
                            )

        # Game tick
        if current_time - last_tick >= tick_interval:
            tick_game(state)
            last_tick = current_time

        # Render
        screen.fill((0, 0, 0))

        # Draw game views (these are still custom pygame rendering)
        draw_player_view(screen, state, Team.RED, red_offset_x, views_offset_y)
        draw_god_view(screen, state, god_offset_x, views_offset_y)
        draw_player_view(screen, state, Team.BLUE, blue_offset_x, views_offset_y)

        # Update slider positions and labels to reflect current state
        if state.tick > 0:
            red_view_tick = state.views[Team.RED].freeze_frame
            if red_view_tick is None:
                red_slider.set_current_value(1.0)
                red_tick_label.set_text(f"t={state.tick}")
            else:
                red_slider.set_current_value(red_view_tick / state.tick)
                red_tick_label.set_text(f"t={red_view_tick}")

            blue_view_tick = state.views[Team.BLUE].freeze_frame
            if blue_view_tick is None:
                blue_slider.set_current_value(1.0)
                blue_tick_label.set_text(f"t={state.tick}")
            else:
                blue_slider.set_current_value(blue_view_tick / state.tick)
                blue_tick_label.set_text(f"t={blue_view_tick}")

        # Plan area layout
        plan_y = slider_y + 30
        plan_box_height = 180
        btn_y = plan_y + plan_box_height + 5

        # Find which team (if any) has a selected unit
        active_team: Team | None = None
        for team in [Team.RED, Team.BLUE]:
            if state.views[team].selected_unit is not None:
                active_team = team
                break

        if active_team is not None:
            view = state.views[active_team]
            selected_unit = view.selected_unit
            assert selected_unit is not None  # for type checker
            team_name = active_team.value
            plan_offset_x = (
                red_offset_x if active_team == Team.RED else blue_offset_x
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
