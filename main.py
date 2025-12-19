from __future__ import annotations

import pygame
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# --- Game Constants ---
GRID_SIZE = 32
TILE_SIZE = 20
MAP_PIXEL_SIZE = GRID_SIZE * TILE_SIZE

# Visibility radius (Manhattan distance) around home base
VISIBILITY_RADIUS = 8


class Team(Enum):
    RED = "RED"
    BLUE = "BLUE"


@dataclass
class Pos:
    x: int
    y: int

    def manhattan_distance(self, other: Pos) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pos):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class MoveOrder:
    target: Pos
    then_return_home: bool = False


@dataclass
class Unit:
    team: Team
    pos: Pos
    order: MoveOrder | None = None

    def home_base(self) -> Pos:
        if self.team == Team.RED:
            return Pos(2, GRID_SIZE // 2)
        else:
            return Pos(GRID_SIZE - 3, GRID_SIZE // 2)


@dataclass
class GameState:
    units: list[Unit] = field(default_factory=list)
    selected_unit: Unit | None = None

    def __post_init__(self) -> None:
        # Create initial units for each team
        red_base = Pos(2, GRID_SIZE // 2)
        blue_base = Pos(GRID_SIZE - 3, GRID_SIZE // 2)

        for i in range(3):
            self.units.append(Unit(Team.RED, Pos(red_base.x, red_base.y - 1 + i)))
            self.units.append(Unit(Team.BLUE, Pos(blue_base.x, blue_base.y - 1 + i)))

    def get_base_pos(self, team: Team) -> Pos:
        if team == Team.RED:
            return Pos(2, GRID_SIZE // 2)
        else:
            return Pos(GRID_SIZE - 3, GRID_SIZE // 2)


def tick_game(state: GameState) -> None:
    """Advance the game by one tick."""
    for unit in state.units:
        if unit.order is None:
            continue

        target = unit.order.target
        dx = 0 if target.x == unit.pos.x else (1 if target.x > unit.pos.x else -1)
        dy = 0 if target.y == unit.pos.y else (1 if target.y > unit.pos.y else -1)

        # Move one step (prefer x then y)
        if dx != 0:
            unit.pos = Pos(unit.pos.x + dx, unit.pos.y)
        elif dy != 0:
            unit.pos = Pos(unit.pos.x, unit.pos.y + dy)

        # Check if arrived at target
        if unit.pos == target:
            if unit.order.then_return_home:
                # Queue return trip
                unit.order = MoveOrder(target=unit.home_base(), then_return_home=False)
            else:
                unit.order = None


# --- Rendering ---

def draw_grid(surface: pygame.Surface, offset_x: int, offset_y: int) -> None:
    """Draw grid lines."""
    for x in range(GRID_SIZE + 1):
        pygame.draw.line(
            surface,
            (50, 50, 50),
            (offset_x + x * TILE_SIZE, offset_y),
            (offset_x + x * TILE_SIZE, offset_y + MAP_PIXEL_SIZE),
        )
    for y in range(GRID_SIZE + 1):
        pygame.draw.line(
            surface,
            (50, 50, 50),
            (offset_x, offset_y + y * TILE_SIZE),
            (offset_x + MAP_PIXEL_SIZE, offset_y + y * TILE_SIZE),
        )


def draw_base(surface: pygame.Surface, pos: Pos, team: Team, offset_x: int, offset_y: int) -> None:
    """Draw a home base marker."""
    color = (200, 50, 50) if team == Team.RED else (50, 50, 200)
    rect = pygame.Rect(
        offset_x + pos.x * TILE_SIZE + 2,
        offset_y + pos.y * TILE_SIZE + 2,
        TILE_SIZE - 4,
        TILE_SIZE - 4,
    )
    pygame.draw.rect(surface, color, rect, 3)


def draw_unit(
    surface: pygame.Surface,
    unit: Unit,
    offset_x: int,
    offset_y: int,
    selected: bool = False,
) -> None:
    """Draw a unit."""
    color = (255, 100, 100) if unit.team == Team.RED else (100, 100, 255)
    center_x = offset_x + unit.pos.x * TILE_SIZE + TILE_SIZE // 2
    center_y = offset_y + unit.pos.y * TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 3

    pygame.draw.circle(surface, color, (center_x, center_y), radius)

    if selected:
        pygame.draw.circle(surface, (255, 255, 0), (center_x, center_y), radius + 3, 2)


def draw_visibility_mask(
    surface: pygame.Surface,
    base_pos: Pos,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw fog of war for tiles outside visibility radius."""
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            tile_pos = Pos(x, y)
            if tile_pos.manhattan_distance(base_pos) > VISIBILITY_RADIUS:
                rect = pygame.Rect(
                    offset_x + x * TILE_SIZE,
                    offset_y + y * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE,
                )
                pygame.draw.rect(surface, (20, 20, 20), rect)


def draw_god_view(
    surface: pygame.Surface,
    state: GameState,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw the god's-eye view of the entire map."""
    # Background
    bg_rect = pygame.Rect(offset_x, offset_y, MAP_PIXEL_SIZE, MAP_PIXEL_SIZE)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    draw_grid(surface, offset_x, offset_y)

    # Bases
    draw_base(surface, state.get_base_pos(Team.RED), Team.RED, offset_x, offset_y)
    draw_base(surface, state.get_base_pos(Team.BLUE), Team.BLUE, offset_x, offset_y)

    # Units
    for unit in state.units:
        draw_unit(surface, unit, offset_x, offset_y, selected=(unit is state.selected_unit))


def draw_player_view(
    surface: pygame.Surface,
    state: GameState,
    team: Team,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw a player's limited view centered on their base."""
    base_pos = state.get_base_pos(team)

    # Background
    bg_rect = pygame.Rect(offset_x, offset_y, MAP_PIXEL_SIZE, MAP_PIXEL_SIZE)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    draw_grid(surface, offset_x, offset_y)

    # Base
    draw_base(surface, base_pos, team, offset_x, offset_y)

    # Units visible from this base
    for unit in state.units:
        if unit.pos.manhattan_distance(base_pos) <= VISIBILITY_RADIUS:
            draw_unit(surface, unit, offset_x, offset_y, selected=(unit is state.selected_unit))

    # Fog of war
    draw_visibility_mask(surface, base_pos, offset_x, offset_y)


def screen_to_grid(mouse_x: int, mouse_y: int, offset_x: int, offset_y: int) -> Pos | None:
    """Convert screen coordinates to grid position."""
    rel_x = mouse_x - offset_x
    rel_y = mouse_y - offset_y

    if 0 <= rel_x < MAP_PIXEL_SIZE and 0 <= rel_y < MAP_PIXEL_SIZE:
        return Pos(rel_x // TILE_SIZE, rel_y // TILE_SIZE)
    return None


def find_unit_at(state: GameState, pos: Pos, team: Team | None = None) -> Unit | None:
    """Find a unit at the given position, optionally filtering by team."""
    for unit in state.units:
        if unit.pos == pos:
            if team is None or unit.team == team:
                return unit
    return None


def main() -> None:
    pygame.init()

    # Layout: RED view | GOD view | BLUE view
    padding = 10
    label_height = 25
    window_width = MAP_PIXEL_SIZE * 3 + padding * 4
    window_height = MAP_PIXEL_SIZE + padding * 2 + label_height

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ant RTS")

    font = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()
    state = GameState()

    # View offsets
    red_offset_x = padding
    god_offset_x = padding * 2 + MAP_PIXEL_SIZE
    blue_offset_x = padding * 3 + MAP_PIXEL_SIZE * 2
    views_offset_y = padding + label_height

    tick_interval = 200  # ms between game ticks
    last_tick = pygame.time.get_ticks()

    running = True
    while running:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Check RED view for unit selection
                grid_pos = screen_to_grid(mx, my, red_offset_x, views_offset_y)
                if grid_pos is not None:
                    red_base = state.get_base_pos(Team.RED)
                    if grid_pos.manhattan_distance(red_base) <= VISIBILITY_RADIUS:
                        unit = find_unit_at(state, grid_pos, Team.RED)
                        if unit is not None:
                            state.selected_unit = unit

                # Check BLUE view for unit selection
                grid_pos = screen_to_grid(mx, my, blue_offset_x, views_offset_y)
                if grid_pos is not None:
                    blue_base = state.get_base_pos(Team.BLUE)
                    if grid_pos.manhattan_distance(blue_base) <= VISIBILITY_RADIUS:
                        unit = find_unit_at(state, grid_pos, Team.BLUE)
                        if unit is not None:
                            state.selected_unit = unit

                # Check GOD view for target selection
                grid_pos = screen_to_grid(mx, my, god_offset_x, views_offset_y)
                if grid_pos is not None and state.selected_unit is not None:
                    # Issue MOVE order
                    state.selected_unit.order = MoveOrder(
                        target=grid_pos,
                        then_return_home=True,
                    )
                    state.selected_unit = None

        # Game tick
        if current_time - last_tick >= tick_interval:
            tick_game(state)
            last_tick = current_time

        # Render
        screen.fill((0, 0, 0))

        # Labels
        red_label = font.render("RED BASE VIEW", True, (255, 100, 100))
        god_label = font.render("GOD'S EYE VIEW", True, (200, 200, 200))
        blue_label = font.render("BLUE BASE VIEW", True, (100, 100, 255))

        screen.blit(red_label, (red_offset_x, padding))
        screen.blit(god_label, (god_offset_x, padding))
        screen.blit(blue_label, (blue_offset_x, padding))

        # Views
        draw_player_view(screen, state, Team.RED, red_offset_x, views_offset_y)
        draw_god_view(screen, state, god_offset_x, views_offset_y)
        draw_player_view(screen, state, Team.BLUE, blue_offset_x, views_offset_y)

        # Selection indicator
        if state.selected_unit is not None:
            team_name = state.selected_unit.team.value
            sel_text = font.render(f"Selected: {team_name} unit - click GOD view to set destination", True, (255, 255, 0))
            screen.blit(sel_text, (window_width // 2 - sel_text.get_width() // 2, window_height - 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
