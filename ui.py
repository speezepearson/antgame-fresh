"""Pygame rendering and UI."""

from __future__ import annotations
from typing import Any
import argparse
import pygame
import pygame_gui

from core import Pos, Region
from mechanics import (
    GameState, Team, BasePresent, UnitPresent, GRID_SIZE, Move, Plan, tick_game,
    Unit, Order, Condition, Action, Interrupt,
    EnemyInRangeCondition, BaseVisibleCondition, PositionReachedCondition,
    FoodConfig,
)


# Rendering Constants
TILE_SIZE = 16


# Plan display helpers
def _describe_condition(condition: Condition[Any]) -> str:
    """Convert a condition to a human-readable string."""
    if isinstance(condition, EnemyInRangeCondition):
        return f"enemy within {condition.distance}"
    elif isinstance(condition, BaseVisibleCondition):
        return "base visible"
    elif isinstance(condition, PositionReachedCondition):
        return f"reached ({condition.position.x}, {condition.position.y})"
    else:
        return "unknown condition"


def _describe_order(order: Order) -> str:
    """Convert an order to a human-readable string."""
    if isinstance(order, Move):
        return f"Move to ({order.target.x}, {order.target.y})"
    else:
        return "Unknown order"


def format_plan(plan: Plan, unit: Unit) -> list[str]:
    """Format a plan as a list of strings for display."""
    lines = []

    # Show current and remaining orders
    if plan.orders:
        lines.append("Orders:")
        for i, order in enumerate(plan.orders):
            prefix = "> " if i == 0 else "  "
            lines.append(f"{prefix}{_describe_order(order)}")
    else:
        lines.append("Orders: (none)")

    # Show interrupts
    if plan.interrupts:
        lines.append("Interrupts:")
        for interrupt in plan.interrupts:
            condition_desc = _describe_condition(interrupt.condition)
            lines.append(f"  If {condition_desc}: <action>")

    return lines


def draw_grid(surface: pygame.Surface, offset_x: int, offset_y: int, grid_width: int, grid_height: int) -> None:
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    for x in range(grid_width + 1):
        pygame.draw.line(
            surface, (50, 50, 50),
            (offset_x + x * TILE_SIZE, offset_y),
            (offset_x + x * TILE_SIZE, offset_y + map_pixel_height),
        )
    for y in range(grid_height + 1):
        pygame.draw.line(
            surface, (50, 50, 50),
            (offset_x, offset_y + y * TILE_SIZE),
            (offset_x + map_pixel_width, offset_y + y * TILE_SIZE),
        )


def draw_base(surface: pygame.Surface, region: Region, team: Team, offset_x: int, offset_y: int) -> None:
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


def draw_food(surface: pygame.Surface, food: dict[Pos, int], offset_x: int, offset_y: int) -> None:
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
                (TILE_SIZE // 2, TILE_SIZE // 3),           # top center
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),       # bottom left
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),   # bottom right
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
            ][:min(count, 5)]

        for dx, dy in positions:
            pygame.draw.circle(surface, (100, 255, 100), (tile_x + dx, tile_y + dy), radius)

        # If more than 5, indicate with a brighter center dot
        if count > 5:
            pygame.draw.circle(surface, (150, 255, 150), (tile_x + TILE_SIZE // 2, tile_y + TILE_SIZE // 2), radius)


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
            surface, unit.team, unit.pos, offset_x, offset_y,
            selected=(unit is state.selected_unit),
        )


def draw_player_view(
    surface: pygame.Surface,
    state: GameState,
    team: Team,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw a player's view of the map at their selected tick."""
    view_t = state.view_tick[team]
    logbook = state.base_logbooks[team]

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
                        draw_base(surface, state.get_base_region(contents.team), contents.team, offset_x, offset_y)
                        drawn_bases.add(contents.team)
        # Second pass: draw food in visible areas
        visible_food = {pos: count for pos, count in state.food.items() if pos in observations}
        draw_food(surface, visible_food, offset_x, offset_y)
        # Third pass: draw units on top
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, UnitPresent):
                    draw_unit_at(surface, contents.team, pos, offset_x, offset_y)


def draw_slider(
    surface: pygame.Surface,
    x: int,
    y: int,
    width: int,
    max_tick: int,
    current_tick: int,
    team: Team,
    is_live: bool,
    font: pygame.font.Font,
) -> tuple[pygame.Rect, pygame.Rect]:
    """Draw a time slider and LIVE button. Return (slider_rect, live_button_rect)."""
    height = 20
    slider_rect = pygame.Rect(x, y, width, height)

    # Background
    pygame.draw.rect(surface, (60, 60, 60), slider_rect)
    pygame.draw.rect(surface, (100, 100, 100), slider_rect, 1)

    # Filled portion
    if max_tick > 0:
        fill_width = int((current_tick / max_tick) * width)
        fill_color = (150, 80, 80) if team == Team.RED else (80, 80, 150)
        fill_rect = pygame.Rect(x, y, fill_width, height)
        pygame.draw.rect(surface, fill_color, fill_rect)

    # Handle
    if max_tick > 0:
        handle_x = x + int((current_tick / max_tick) * width)
    else:
        handle_x = x
    pygame.draw.line(surface, (255, 255, 255), (handle_x, y), (handle_x, y + height), 2)

    # Label
    label = font.render(f"t={current_tick}", True, (200, 200, 200))
    surface.blit(label, (x + width + 5, y + 2))

    # LIVE button
    live_btn_x = x + width + 50
    live_btn_rect = pygame.Rect(live_btn_x, y, 35, height)
    btn_color = (50, 150, 50) if is_live else (80, 80, 80)
    pygame.draw.rect(surface, btn_color, live_btn_rect)
    pygame.draw.rect(surface, (150, 150, 150), live_btn_rect, 1)
    live_label = font.render("LIVE", True, (255, 255, 255) if is_live else (150, 150, 150))
    surface.blit(live_label, (live_btn_x + 4, y + 4))

    return slider_rect, live_btn_rect


def screen_to_grid(mouse_x: int, mouse_y: int, offset_x: int, offset_y: int, grid_width: int, grid_height: int) -> Pos | None:
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


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ant RTS Game')
    parser.add_argument('--width', type=int, default=GRID_SIZE,
                        help='Width of the grid (default: 32)')
    parser.add_argument('--height', type=int, default=GRID_SIZE,
                        help='Height of the grid (default: 32)')
    parser.add_argument('--food-scale', type=float, default=10.0,
                        help='Perlin noise scale for food generation (default: 10.0)')
    parser.add_argument('--food-max-prob', type=float, default=0.1,
                        help='Maximum probability of food in a cell (default: 0.1)')
    parser.add_argument('--food-seed', type=int, default=None,
                        help='Random seed for food generation (default: random)')
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
    state = GameState(grid_width=args.width, grid_height=args.height, food_config=food_config)
    map_pixel_size = state.grid_width * TILE_SIZE

    window_width = map_pixel_size * 3 + padding * 4
    window_height = map_pixel_size + padding * 2 + label_height + slider_height + plan_area_height

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ant RTS")

    font = pygame.font.Font(None, 20)

    # Initialize pygame_gui manager
    ui_manager = pygame_gui.UIManager((window_width, window_height))

    clock = pygame.time.Clock()

    # Plan text box (will be created when needed)
    plan_text_box: pygame_gui.elements.UITextBox | None = None
    last_plan_html: str | None = None  # Track last plan content to avoid unnecessary updates

    red_offset_x = padding
    god_offset_x = padding * 2 + map_pixel_size
    blue_offset_x = padding * 3 + map_pixel_size * 2
    views_offset_y = padding + label_height

    tick_interval = 200
    last_tick = pygame.time.get_ticks()

    # Slider and button rects (will be set during render)
    red_slider_rect = pygame.Rect(0, 0, 0, 0)
    blue_slider_rect = pygame.Rect(0, 0, 0, 0)
    red_live_rect = pygame.Rect(0, 0, 0, 0)
    blue_live_rect = pygame.Rect(0, 0, 0, 0)
    dragging_slider: Team | None = None

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            # Process pygame_gui events
            ui_manager.process_events(event)

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Skip manual handling if click is on a UI element (like the scrollbar)
                if plan_text_box is not None and plan_text_box.rect.collidepoint(mx, my):
                    continue

                # Check LIVE buttons
                if red_live_rect.collidepoint(mx, my):
                    state.view_live[Team.RED] = True
                    state.view_tick[Team.RED] = state.tick
                elif blue_live_rect.collidepoint(mx, my):
                    state.view_live[Team.BLUE] = True
                    state.view_tick[Team.BLUE] = state.tick
                # Check sliders
                elif red_slider_rect.collidepoint(mx, my):
                    dragging_slider = Team.RED
                    state.view_live[Team.RED] = False
                elif blue_slider_rect.collidepoint(mx, my):
                    dragging_slider = Team.BLUE
                    state.view_live[Team.BLUE] = False
                # Check Issue Plan button
                elif issue_plan_rect is not None and issue_plan_rect.collidepoint(mx, my):
                    if state.working_plan is not None and state.working_plan.orders and state.selected_unit is not None:
                        # Assign working plan to selected unit
                        state.selected_unit.plan = state.working_plan
                        state.selected_unit = None
                        state.working_plan = None
                # Check Clear button
                elif clear_plan_rect is not None and clear_plan_rect.collidepoint(mx, my):
                    # Reset working plan
                    state.working_plan = Plan()
                else:
                    # Check RED player view for unit selection or target
                    grid_pos = screen_to_grid(mx, my, red_offset_x, views_offset_y, state.grid_width, state.grid_height)
                    if grid_pos is not None and state.view_tick[Team.RED] == state.tick:
                        if state.selected_unit is not None and state.selected_unit.team == Team.RED:
                            # Append Move order to working plan
                            if state.working_plan is None:
                                state.working_plan = Plan()
                            state.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base(state, grid_pos, Team.RED)
                            if unit is not None:
                                state.selected_unit = unit
                                # Initialize working plan when selecting a unit
                                state.working_plan = Plan()

                    # Check BLUE player view for unit selection or target
                    grid_pos = screen_to_grid(mx, my, blue_offset_x, views_offset_y, state.grid_width, state.grid_height)
                    if grid_pos is not None and state.view_tick[Team.BLUE] == state.tick:
                        if state.selected_unit is not None and state.selected_unit.team == Team.BLUE:
                            # Append Move order to working plan
                            if state.working_plan is None:
                                state.working_plan = Plan()
                            state.working_plan.orders.append(Move(target=grid_pos))
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base(state, grid_pos, Team.BLUE)
                            if unit is not None:
                                state.selected_unit = unit
                                # Initialize working plan when selecting a unit
                                state.working_plan = Plan()

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Only handle if not on UI element
                mx, my = event.pos
                if plan_text_box is None or not plan_text_box.rect.collidepoint(mx, my):
                    dragging_slider = None

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                # Skip if on UI element
                if plan_text_box is not None and plan_text_box.rect.collidepoint(mx, my):
                    continue

                if dragging_slider is not None:
                    if dragging_slider == Team.RED:
                        rel_x = mx - red_slider_rect.x
                        pct = max(0, min(1, rel_x / red_slider_rect.width))
                        state.view_tick[Team.RED] = int(pct * state.tick)
                    else:
                        rel_x = mx - blue_slider_rect.x
                        pct = max(0, min(1, rel_x / blue_slider_rect.width))
                        state.view_tick[Team.BLUE] = int(pct * state.tick)

        # Game tick
        if current_time - last_tick >= tick_interval:
            tick_game(state)
            last_tick = current_time
            # Auto-advance sliders only if in live mode
            if state.view_live[Team.RED]:
                state.view_tick[Team.RED] = state.tick
            if state.view_live[Team.BLUE]:
                state.view_tick[Team.BLUE] = state.tick

        # Render
        screen.fill((0, 0, 0))

        # Labels
        red_label = font.render("RED'S MAP", True, (255, 100, 100))
        god_label = font.render("GOD'S EYE VIEW", True, (200, 200, 200))
        blue_label = font.render("BLUE'S MAP", True, (100, 100, 255))

        screen.blit(red_label, (red_offset_x, padding))
        screen.blit(god_label, (god_offset_x, padding))
        screen.blit(blue_label, (blue_offset_x, padding))

        # Views
        draw_player_view(screen, state, Team.RED, red_offset_x, views_offset_y)
        draw_god_view(screen, state, god_offset_x, views_offset_y)
        draw_player_view(screen, state, Team.BLUE, blue_offset_x, views_offset_y)

        # Sliders
        slider_y = views_offset_y + map_pixel_size + 5
        slider_width = map_pixel_size - 100

        red_slider_rect, red_live_rect = draw_slider(
            screen, red_offset_x, slider_y, slider_width,
            state.tick, state.view_tick[Team.RED], Team.RED,
            state.view_live[Team.RED], font,
        )
        blue_slider_rect, blue_live_rect = draw_slider(
            screen, blue_offset_x, slider_y, slider_width,
            state.tick, state.view_tick[Team.BLUE], Team.BLUE,
            state.view_live[Team.BLUE], font,
        )

        # Selection indicator and plan display
        issue_plan_rect: pygame.Rect | None = None
        clear_plan_rect: pygame.Rect | None = None
        if state.selected_unit is not None:
            # Determine which player's map to show the plan under
            team_name = state.selected_unit.team.value
            plan_offset_x = red_offset_x if state.selected_unit.team == Team.RED else blue_offset_x

            # Show selection indicator in God's Eye view area
            sel_text = font.render(
                f"Selected: {team_name} unit - click {team_name}'s map to add waypoints",
                True, (255, 255, 0),
            )
            screen.blit(sel_text, (god_offset_x, slider_y))

            # Display the working plan in a scrollable text box
            plan_y = slider_y + 30  # Start below the slider
            plan_box_height = 180  # Fixed height for scrollable area (doubled)
            btn_y = plan_y + plan_box_height + 5  # Position buttons below the text box

            if state.working_plan is not None:
                plan_lines = format_plan(state.working_plan, state.selected_unit)
                plan_html = "<br>".join(plan_lines)

                # Create or update the plan text box
                text_box_rect = pygame.Rect(plan_offset_x, plan_y, map_pixel_size, plan_box_height)

                # Only recreate or update if position changed or content changed
                if plan_text_box is None or plan_text_box.relative_rect != text_box_rect:
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

            # Draw "Issue Plan" and "Clear" buttons below the plan
            btn_height = 20

            # Issue Plan button
            issue_plan_rect = pygame.Rect(plan_offset_x, btn_y, 80, btn_height)
            btn_color = (50, 150, 50) if state.working_plan and state.working_plan.orders else (80, 80, 80)
            pygame.draw.rect(screen, btn_color, issue_plan_rect)
            pygame.draw.rect(screen, (150, 150, 150), issue_plan_rect, 1)
            issue_text = font.render("Issue Plan", True, (255, 255, 255))
            screen.blit(issue_text, (issue_plan_rect.x + 5, issue_plan_rect.y + 4))

            # Clear button
            clear_plan_rect = pygame.Rect(plan_offset_x + 90, btn_y, 50, btn_height)
            pygame.draw.rect(screen, (150, 50, 50), clear_plan_rect)
            pygame.draw.rect(screen, (150, 150, 150), clear_plan_rect, 1)
            clear_text = font.render("Clear", True, (255, 255, 255))
            screen.blit(clear_text, (clear_plan_rect.x + 8, clear_plan_rect.y + 4))
        elif plan_text_box is not None:
            # No selected unit, remove the text box
            plan_text_box.kill()
            plan_text_box = None

        # Update UI manager
        ui_manager.update(time_delta)

        # Draw UI manager
        ui_manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
