"""Pygame rendering and UI."""

from __future__ import annotations
import pygame

from core import Pos, Region
from mechanics import GameState, Team, BasePresent, UnitPresent, GRID_SIZE, MoveOrder, tick_game


# Rendering Constants
TILE_SIZE = 16
MAP_PIXEL_SIZE = GRID_SIZE * TILE_SIZE


def draw_grid(surface: pygame.Surface, offset_x: int, offset_y: int) -> None:
    for x in range(GRID_SIZE + 1):
        pygame.draw.line(
            surface, (50, 50, 50),
            (offset_x + x * TILE_SIZE, offset_y),
            (offset_x + x * TILE_SIZE, offset_y + MAP_PIXEL_SIZE),
        )
    for y in range(GRID_SIZE + 1):
        pygame.draw.line(
            surface, (50, 50, 50),
            (offset_x, offset_y + y * TILE_SIZE),
            (offset_x + MAP_PIXEL_SIZE, offset_y + y * TILE_SIZE),
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


def draw_god_view(
    surface: pygame.Surface,
    state: GameState,
    offset_x: int,
    offset_y: int,
) -> None:
    bg_rect = pygame.Rect(offset_x, offset_y, MAP_PIXEL_SIZE, MAP_PIXEL_SIZE)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    draw_grid(surface, offset_x, offset_y)
    draw_base(surface, state.get_base_region(Team.RED), Team.RED, offset_x, offset_y)
    draw_base(surface, state.get_base_region(Team.BLUE), Team.BLUE, offset_x, offset_y)

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

    bg_rect = pygame.Rect(offset_x, offset_y, MAP_PIXEL_SIZE, MAP_PIXEL_SIZE)
    pygame.draw.rect(surface, (20, 20, 20), bg_rect)

    draw_grid(surface, offset_x, offset_y)

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
        draw_grid(surface, offset_x, offset_y)

        # Draw observed contents (bases first, then units on top)
        drawn_bases = set()
        # First pass: draw bases
        for pos, contents_list in observations.items():
            for contents in contents_list:
                if isinstance(contents, BasePresent):
                    if contents.team not in drawn_bases:
                        draw_base(surface, state.get_base_region(contents.team), contents.team, offset_x, offset_y)
                        drawn_bases.add(contents.team)
        # Second pass: draw units on top
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


def screen_to_grid(mouse_x: int, mouse_y: int, offset_x: int, offset_y: int) -> Pos | None:
    rel_x = mouse_x - offset_x
    rel_y = mouse_y - offset_y

    if 0 <= rel_x < MAP_PIXEL_SIZE and 0 <= rel_y < MAP_PIXEL_SIZE:
        return Pos(rel_x // TILE_SIZE, rel_y // TILE_SIZE)
    return None


def find_unit_at_base(state: GameState, pos: Pos, team: Team):
    """Find a unit of the given team at pos, only if inside their base region."""
    base_region = state.get_base_region(team)
    for unit in state.units:
        if unit.team == team and unit.pos == pos:
            if base_region.contains(unit.pos):
                return unit
    return None


def main() -> None:
    pygame.init()

    # Layout: RED view (with slider) | GOD view | BLUE view (with slider)
    padding = 10
    label_height = 25
    slider_height = 30
    window_width = MAP_PIXEL_SIZE * 3 + padding * 4
    window_height = MAP_PIXEL_SIZE + padding * 2 + label_height + slider_height

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ant RTS")

    font = pygame.font.Font(None, 20)

    clock = pygame.time.Clock()
    state = GameState()

    red_offset_x = padding
    god_offset_x = padding * 2 + MAP_PIXEL_SIZE
    blue_offset_x = padding * 3 + MAP_PIXEL_SIZE * 2
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

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
                else:
                    # Check RED player view for unit selection or target
                    grid_pos = screen_to_grid(mx, my, red_offset_x, views_offset_y)
                    if grid_pos is not None and state.view_tick[Team.RED] == state.tick:
                        if state.selected_unit is not None and state.selected_unit.team == Team.RED:
                            # Issue move order
                            state.selected_unit.order = MoveOrder(
                                target=grid_pos,
                                then_return_home=True,
                            )
                            state.selected_unit = None
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base(state, grid_pos, Team.RED)
                            if unit is not None:
                                state.selected_unit = unit

                    # Check BLUE player view for unit selection or target
                    grid_pos = screen_to_grid(mx, my, blue_offset_x, views_offset_y)
                    if grid_pos is not None and state.view_tick[Team.BLUE] == state.tick:
                        if state.selected_unit is not None and state.selected_unit.team == Team.BLUE:
                            # Issue move order
                            state.selected_unit.order = MoveOrder(
                                target=grid_pos,
                                then_return_home=True,
                            )
                            state.selected_unit = None
                        else:
                            # Try to select a unit
                            unit = find_unit_at_base(state, grid_pos, Team.BLUE)
                            if unit is not None:
                                state.selected_unit = unit

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_slider = None

            elif event.type == pygame.MOUSEMOTION and dragging_slider is not None:
                mx, my = event.pos
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
        slider_y = views_offset_y + MAP_PIXEL_SIZE + 5
        slider_width = MAP_PIXEL_SIZE - 100

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

        # Selection indicator
        if state.selected_unit is not None:
            team_name = state.selected_unit.team.value
            sel_text = font.render(
                f"Selected: {team_name} unit - click {team_name}'s map to set destination",
                True, (255, 255, 0),
            )
            screen.blit(sel_text, (god_offset_x, slider_y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
