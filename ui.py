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
from client import GameClient, LocalClient, RemoteClient
from mechanics import (
    Empty,
    FoodInRangeCondition,
    FoodPresent,
    GameState,
    IdleCondition,
    MoveHomeAction,
    MoveThereAction,
    ResumeAction,
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


# Metadata for building interrupt editor UI
@dataclass
class ConditionType:
    """Metadata about a condition type for UI purposes."""

    name: str
    factory: Any  # Callable that creates the condition
    param_specs: dict[str, type]  # parameter name -> type
    return_type: type  # What type this condition returns when triggered


@dataclass
class ActionType:
    """Metadata about an action type for UI purposes."""

    name: str
    factory: Any  # Callable that creates the action
    param_specs: dict[str, type]  # parameter name -> type
    accepts_type: type  # What type this action accepts


# Available condition types
CONDITION_TYPES = [
    ConditionType(
        name="Enemy in Range",
        factory=EnemyInRangeCondition,
        param_specs={"distance": int},
        return_type=Pos,
    ),
    ConditionType(
        name="Food in Range",
        factory=FoodInRangeCondition,
        param_specs={"distance": int},
        return_type=Pos,
    ),
    ConditionType(
        name="Idle",
        factory=IdleCondition,
        param_specs={},
        return_type=type(True),
    ),
    ConditionType(
        name="Base Visible",
        factory=BaseVisibleCondition,
        param_specs={},
        return_type=Pos,
    ),
    ConditionType(
        name="Position Reached",
        factory=PositionReachedCondition,
        param_specs={"position": Pos},
        return_type=Pos,
    ),
]

# Available action types
ACTION_TYPES = [
    ActionType(
        name="Move There",
        factory=MoveThereAction,
        param_specs={},
        accepts_type=Pos,
    ),
    ActionType(
        name="Move Home",
        factory=MoveHomeAction,
        param_specs={},
        accepts_type=object,
    ),
    ActionType(
        name="Resume",
        factory=ResumeAction,
        param_specs={},
        accepts_type=object,
    ),
]


def get_compatible_actions(condition_return_type: type) -> list[ActionType]:
    """Get actions compatible with a given condition's return type."""
    compatible = []
    for action_type in ACTION_TYPES:
        # object accepts any type
        if action_type.accepts_type == object:
            compatible.append(action_type)
        # Check if types match (simplified type checking)
        elif condition_return_type == action_type.accepts_type:
            compatible.append(action_type)
    return compatible


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
    interrupt_checkboxes: dict[str, pygame_gui.elements.UICheckBox] = field(default_factory=dict)
    interrupt_labels: dict[str, pygame_gui.elements.UILabel] = field(default_factory=dict)
    interrupt_container: pygame_gui.elements.UIScrollingContainer | None = None


@dataclass
class InterruptLibraryControls:
    """UI controls for interrupt library management."""

    window: pygame_gui.elements.UIWindow
    interrupt_rows: dict[str, tuple[pygame_gui.elements.UIButton, ...]]  # name -> (edit, delete, rename, dup buttons)
    scroll_container: pygame_gui.elements.UIScrollingContainer
    new_btn: pygame_gui.elements.UIButton
    close_btn: pygame_gui.elements.UIButton


@dataclass
class InterruptEditorControls:
    """UI controls for editing/creating an interrupt."""

    window: pygame_gui.elements.UIWindow
    name_entry: pygame_gui.elements.UITextEntryLine
    condition_dropdown: pygame_gui.elements.UIDropDownMenu
    condition_param_labels: dict[str, pygame_gui.elements.UILabel]
    condition_param_entries: dict[str, pygame_gui.elements.UITextEntryLine]
    action_list: pygame_gui.elements.UITextBox
    add_action_dropdown: pygame_gui.elements.UIDropDownMenu
    add_action_btn: pygame_gui.elements.UIButton
    remove_action_btn: pygame_gui.elements.UIButton
    save_btn: pygame_gui.elements.UIButton
    cancel_btn: pygame_gui.elements.UIButton
    selected_condition_type: ConditionType | None = None
    selected_actions: list[tuple[ActionType, dict[str, Any]]] = field(default_factory=list)
    original_interrupt_name: str = ""


@dataclass
class PlayerView:
    knowledge: PlayerKnowledge
    tick_controls: TickControls
    plan_controls: PlanControls | None = None
    library_controls: InterruptLibraryControls | None = None
    editor_controls: InterruptEditorControls | None = None
    freeze_frame: Timestamp | None = None
    selected_unit_id: UnitId | None = None
    working_plan: Plan | None = None
    last_click_time: int = 0
    last_click_pos: Pos | None = None


@dataclass
class GameContext:
    """Container for all game state and UI elements."""

    client: GameClient
    grid_width: int
    grid_height: int
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
    library_buttons: dict[Team, pygame_gui.elements.UIButton] = field(default_factory=dict)


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
    state: GameState | None,
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
) -> None:
    """Draw the god's-eye view showing the complete game state.

    If state is None (client mode), draws a placeholder message instead.
    """
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    bg_rect = pygame.Rect(offset_x, offset_y, map_pixel_width, map_pixel_height)
    pygame.draw.rect(surface, (30, 30, 30), bg_rect)

    if state is None:
        # Client mode - no god view available
        # Draw a message indicating god view is not available
        font = pygame.font.Font(None, 24)
        text = font.render(
            "God view not available in client mode", True, (100, 100, 100)
        )
        text_rect = text.get_rect(
            center=(offset_x + map_pixel_width // 2, offset_y + map_pixel_height // 2)
        )
        surface.blit(text, text_rect)
        return

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


CELL_BRIGHTNESS_HALFLIFE = 100
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
    pygame.draw.rect(surface, (0, 0, 0), bg_rect)

    # Get observations for this timestamp
    cur_observations = logbook.get(view_t, {})

    # Draw cells with gradient tinting based on observation age
    for pos, (last_observed_tick, _) in view.knowledge.last_observations.items():
        age = view_t - last_observed_tick
        # Exponential decay: 80 * 2^-(age/50)
        brightness = int(80 * (2 ** (-(age / CELL_BRIGHTNESS_HALFLIFE))))
        brightness = max(0, min(80, brightness))  # Clamp to [0, 80]

        rect = pygame.Rect(
            offset_x + pos.x * TILE_SIZE,
            offset_y + pos.y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(surface, (brightness, brightness, brightness), rect)

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


def find_unit_at_base(client: GameClient, pos: Pos, team: Team) -> Unit | None:
    """Find a unit of the given team at pos, only if inside their base region."""
    base_region = client.get_base_region(team)
    knowledge = client.get_player_knowledge(team, client.get_current_tick())

    # Check units we know about from our observations
    for unit_id, (timestamp, unit) in knowledge.last_seen.items():
        if unit.team == team and unit.pos == pos:
            if base_region.contains(unit.pos):
                return unit

    return None

attack_nearby_enemy_interrupt = Interrupt(
    condition=EnemyInRangeCondition(distance=2),
    actions=[MoveThereAction()],
)
flee_enemy_interrupt = Interrupt(
    condition=EnemyInRangeCondition(distance=2),
    actions=[MoveHomeAction()],
)
get_food_interrupt = Interrupt(
    condition=FoodInRangeCondition(distance=2),
    actions=[MoveThereAction(), ResumeAction()],
)
go_home_when_done_interrupt = Interrupt(
    condition=IdleCondition(),
    actions=[MoveHomeAction()],
)


def make_default_interrupts() -> list[Interrupt[Any]]:
    return [
        attack_nearby_enemy_interrupt if random.random() < 0.5 else flee_enemy_interrupt,
        get_food_interrupt,
        go_home_when_done_interrupt,
    ]


def initialize_interrupt_library() -> dict[str, Interrupt[Any]]:
    """Create the initial interrupt library with default interrupts."""
    return {
        "attack_nearby_enemy": attack_nearby_enemy_interrupt,
        "flee_enemy": flee_enemy_interrupt,
        "get_food": get_food_interrupt,
        "go_home_when_done": go_home_when_done_interrupt,
    }


def get_default_plan_interrupts(library: dict[str, Interrupt[Any]]) -> list[Interrupt[Any]]:
    """Get the default interrupts to use for new plans from the library."""
    # Use the interrupts from the library if available
    default_names = ["get_food", "go_home_when_done"]
    interrupts = []
    for name in default_names:
        if name in library:
            interrupts.append(library[name])
    # Add attack or flee (pick one from library)
    if "attack_nearby_enemy" in library:
        interrupts.insert(0, library["attack_nearby_enemy"])
    elif "flee_enemy" in library:
        interrupts.insert(0, library["flee_enemy"])
    return interrupts if interrupts else []


def format_interrupt_library(library: dict[str, Interrupt[Any]]) -> str:
    """Format an interrupt library as HTML for display."""
    if not library:
        return "<b>No interrupts in library</b>"

    lines = ["<b>Interrupt Library:</b><br>"]
    for name, interrupt in sorted(library.items()):
        condition_desc = interrupt.condition.description
        actions_desc = "; ".join([action.description for action in interrupt.actions])
        lines.append(f"<br><b>{name}</b>: If {condition_desc}: {actions_desc}")

    return "".join(lines)


def create_simple_interrupt_editor(
    ctx: GameContext, team: Team, interrupt_name: str
) -> InterruptEditorControls:
    """Create a simple interrupt editor dialog."""
    window_width = 500
    window_height = 600
    window = pygame_gui.elements.UIWindow(
        rect=pygame.Rect(150, 50, window_width, window_height),
        manager=ctx.ui_manager,
        window_display_title=f"Edit Interrupt: {interrupt_name}",
    )

    view = ctx.views[team]
    interrupt = view.knowledge.interrupt_library.get(interrupt_name)

    if interrupt is None:
        raise ValueError(f"Interrupt '{interrupt_name}' not found in library")

    y_offset = 10

    # Name entry
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(10, y_offset, 100, 25),
        text="Name:",
        manager=ctx.ui_manager,
        container=window,
    )
    name_entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(120, y_offset, window_width - 140, 25),
        manager=ctx.ui_manager,
        container=window,
    )
    name_entry.set_text(interrupt_name)
    y_offset += 40

    # Condition dropdown
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(10, y_offset, 100, 25),
        text="Condition:",
        manager=ctx.ui_manager,
        container=window,
    )
    condition_options = [ct.name for ct in CONDITION_TYPES]

    # Find current condition type
    current_condition = None
    selected_condition_type = None
    for ct in CONDITION_TYPES:
        if type(interrupt.condition) == ct.factory:
            current_condition = ct.name
            selected_condition_type = ct
            break
    if current_condition is None:
        current_condition = condition_options[0]
        selected_condition_type = CONDITION_TYPES[0]

    condition_dropdown = pygame_gui.elements.UIDropDownMenu(
        options_list=list(condition_options),
        starting_option=current_condition,
        relative_rect=pygame.Rect(120, y_offset, window_width - 140, 30),
        manager=ctx.ui_manager,
        container=window,
    )
    y_offset += 40

    # Condition parameters
    condition_param_labels: dict[str, pygame_gui.elements.UILabel] = {}
    condition_param_entries: dict[str, pygame_gui.elements.UITextEntryLine] = {}

    if selected_condition_type and selected_condition_type.param_specs:
        for param_name, param_type in selected_condition_type.param_specs.items():
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(30, y_offset, 100, 25),
                text=f"{param_name}:",
                manager=ctx.ui_manager,
                container=window,
            )

            # Get current value from interrupt
            current_value = ""
            if hasattr(interrupt.condition, param_name):
                val = getattr(interrupt.condition, param_name)
                if isinstance(val, Pos):
                    current_value = f"{val.x},{val.y}"
                else:
                    current_value = str(val)

            entry = pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect(140, y_offset, window_width - 160, 25),
                manager=ctx.ui_manager,
                container=window,
            )
            entry.set_text(current_value)

            condition_param_labels[param_name] = label
            condition_param_entries[param_name] = entry
            y_offset += 35

    # Action list display
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(10, y_offset, 100, 25),
        text="Actions:",
        manager=ctx.ui_manager,
        container=window,
    )
    y_offset += 30

    action_list = pygame_gui.elements.UITextBox(
        html_text="",
        relative_rect=pygame.Rect(10, y_offset, window_width - 20, 150),
        manager=ctx.ui_manager,
        container=window,
    )
    y_offset += 160

    # Action controls
    compatible_actions = get_compatible_actions(selected_condition_type.return_type) if selected_condition_type else ACTION_TYPES
    action_options = [at.name for at in compatible_actions]

    add_action_dropdown = pygame_gui.elements.UIDropDownMenu(
        options_list=list(action_options) if action_options else ["(none)"],
        starting_option=action_options[0] if action_options else "(none)",
        relative_rect=pygame.Rect(10, y_offset, 200, 30),
        manager=ctx.ui_manager,
        container=window,
    )

    add_action_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(220, y_offset, 100, 25),
        text="Add ➕",
        manager=ctx.ui_manager,
        container=window,
    )

    remove_action_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(330, y_offset, 120, 25),
        text="Remove ❌",
        manager=ctx.ui_manager,
        container=window,
    )
    y_offset += 40

    # Save and cancel buttons
    save_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(10, window_height - 60, 80, 30),
        text="Save",
        manager=ctx.ui_manager,
        container=window,
    )

    cancel_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(100, window_height - 60, 80, 30),
        text="Cancel",
        manager=ctx.ui_manager,
        container=window,
    )

    # Convert current actions to (ActionType, params) tuples
    selected_actions: list[tuple[ActionType, dict[str, Any]]] = []
    for action in interrupt.actions:
        for at in ACTION_TYPES:
            if type(action) == at.factory:
                selected_actions.append((at, {}))
                break

    return InterruptEditorControls(
        window=window,
        name_entry=name_entry,
        condition_dropdown=condition_dropdown,
        condition_param_labels=condition_param_labels,
        condition_param_entries=condition_param_entries,
        action_list=action_list,
        add_action_dropdown=add_action_dropdown,
        add_action_btn=add_action_btn,
        remove_action_btn=remove_action_btn,
        save_btn=save_btn,
        cancel_btn=cancel_btn,
        selected_condition_type=selected_condition_type,
        selected_actions=selected_actions,
        original_interrupt_name=interrupt_name,
    )


def create_interrupt_library_window(
    ctx: GameContext, team: Team
) -> InterruptLibraryControls:
    """Create a window showing the interrupt library for a team."""
    window_width = 700
    window_height = 500
    window = pygame_gui.elements.UIWindow(
        rect=pygame.Rect(100, 100, window_width, window_height),
        manager=ctx.ui_manager,
        window_display_title=f"{team.value} Interrupt Library",
    )

    view = ctx.views[team]

    # Scrolling container for interrupt rows
    scroll_container = pygame_gui.elements.UIScrollingContainer(
        relative_rect=pygame.Rect(10, 10, window_width - 20, window_height - 80),
        manager=ctx.ui_manager,
        container=window,
    )

    # Create rows for each interrupt
    interrupt_rows: dict[str, tuple[pygame_gui.elements.UIButton, ...]] = {}
    y_offset = 0
    row_height = 60

    for name in sorted(view.knowledge.interrupt_library.keys()):
        interrupt = view.knowledge.interrupt_library[name]

        # Interrupt name and description
        condition_desc = interrupt.condition.description
        actions_desc = "; ".join([action.description for action in interrupt.actions])
        text = f"<b>{name}</b><br><font size=2>When {condition_desc}: {actions_desc}</font>"

        pygame_gui.elements.UITextBox(
            html_text=text,
            relative_rect=pygame.Rect(5, y_offset, window_width - 250, row_height - 5),
            manager=ctx.ui_manager,
            container=scroll_container,
        )

        # Action buttons with emojis
        btn_width = 50
        btn_x = window_width - 240

        edit_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(btn_x, y_offset + 15, btn_width, 30),
            text="✏️",
            manager=ctx.ui_manager,
            container=scroll_container,
            tool_tip_text="Edit",
        )
        btn_x += btn_width + 5

        duplicate_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(btn_x, y_offset + 15, btn_width, 30),
            text="📋",
            manager=ctx.ui_manager,
            container=scroll_container,
            tool_tip_text="Duplicate",
        )
        btn_x += btn_width + 5

        rename_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(btn_x, y_offset + 15, btn_width, 30),
            text="✍️",
            manager=ctx.ui_manager,
            container=scroll_container,
            tool_tip_text="Rename",
        )
        btn_x += btn_width + 5

        delete_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(btn_x, y_offset + 15, btn_width, 30),
            text="🗑️",
            manager=ctx.ui_manager,
            container=scroll_container,
            tool_tip_text="Delete",
        )

        interrupt_rows[name] = (edit_btn, duplicate_btn, rename_btn, delete_btn)
        y_offset += row_height

    # Set scrollable area height
    scroll_container.set_scrollable_area_dimensions((window_width - 40, max(y_offset, window_height - 100)))

    # Bottom buttons
    btn_y = window_height - 60

    new_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(10, btn_y, 100, 30),
        text="➕ New",
        manager=ctx.ui_manager,
        container=window,
    )

    close_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(window_width - 110, btn_y, 100, 30),
        text="Close",
        manager=ctx.ui_manager,
        container=window,
    )

    return InterruptLibraryControls(
        window=window,
        interrupt_rows=interrupt_rows,
        scroll_container=scroll_container,
        new_btn=new_btn,
        close_btn=close_btn,
    )


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

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    pygame.init()

    # Layout: RED view (with slider) | GOD view | BLUE view (with slider)
    padding = 10
    label_height = 25
    slider_height = 30
    plan_area_height = 240  # Space for plan display and buttons below player maps

    # Create the game state and client based on mode
    if args.mode == "client":
        # Client mode: create RemoteClient first, get dimensions from it
        client_team = Team[args.team]
        from client import RemoteClient

        temp_client = RemoteClient(url=args.url, team=client_team)
        # Fetch initial knowledge to get grid dimensions
        initial_knowledge = temp_client.get_player_knowledge(client_team, Timestamp(0))
        grid_width = initial_knowledge.grid_width
        grid_height = initial_knowledge.grid_height
        state = None  # No local state in client mode
    else:
        # Local or server mode: create GameState
        food_config = FoodConfig(
            scale=args.food_scale,
            max_prob=args.food_max_prob,
        )
        state = make_game(
            grid_width=args.width, grid_height=args.height, food_config=food_config
        )
        grid_width = state.grid_width
        grid_height = state.grid_height

    map_pixel_size = grid_width * TILE_SIZE

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

    # Create library buttons for each team
    red_library_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(red_offset_x, padding, 60, 20),
        text="Library",
        manager=ui_manager,
    )
    blue_library_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(blue_offset_x, padding, 60, 20),
        text="Library",
        manager=ui_manager,
    )

    # Create time sliders for RED team
    red_tick_controls = TickControls(
        slider=pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(red_offset_x, slider_y, slider_width, 20),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=ui_manager,
        ),
        tick_label=pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                red_offset_x + slider_width + 5, slider_y, 45, 20
            ),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                red_offset_x + slider_width + 50, slider_y, 50, 20
            ),
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
            relative_rect=pygame.Rect(
                blue_offset_x + slider_width + 5, slider_y, 45, 20
            ),
            text="t=0",
            manager=ui_manager,
        ),
        live_btn=pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                blue_offset_x + slider_width + 50, slider_y, 50, 20
            ),
            text="LIVE",
            manager=ui_manager,
        ),
    )

    tick_interval = 200
    last_tick = pygame.time.get_ticks()

    # Create views and client based on mode
    client: GameClient
    if args.mode == "client":
        # Client mode: only create view for the client's team
        # Reuse the temp_client we created earlier
        client = temp_client
        client_team = Team[args.team]

        # Create a placeholder knowledge that will be updated by fetches
        # Initialize interrupt library for the initial knowledge
        initial_knowledge.interrupt_library = initialize_interrupt_library()

        if client_team == Team.RED:
            red_view = PlayerView(
                knowledge=initial_knowledge,
                tick_controls=red_tick_controls,
            )
            # Create a dummy blue view (won't be visible)
            dummy_blue_knowledge = PlayerKnowledge(
                team=Team.BLUE,
                grid_width=grid_width,
                grid_height=grid_height,
                tick=Timestamp(0),
            )
            dummy_blue_knowledge.interrupt_library = initialize_interrupt_library()
            blue_view = PlayerView(
                knowledge=dummy_blue_knowledge,
                tick_controls=blue_tick_controls,
            )
        else:
            # Create a dummy red view (won't be visible)
            dummy_red_knowledge = PlayerKnowledge(
                team=Team.RED,
                grid_width=grid_width,
                grid_height=grid_height,
                tick=Timestamp(0),
            )
            dummy_red_knowledge.interrupt_library = initialize_interrupt_library()
            red_view = PlayerView(
                knowledge=dummy_red_knowledge,
                tick_controls=red_tick_controls,
            )
            blue_view = PlayerView(
                knowledge=initial_knowledge,
                tick_controls=blue_tick_controls,
            )
    else:
        # Local or server mode: create views for both teams
        assert state is not None, "State should not be None in local/server mode"

        red_knowledge = PlayerKnowledge(
            team=Team.RED,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.tick,
        )
        red_knowledge.interrupt_library = initialize_interrupt_library()
        red_view = PlayerView(
            knowledge=red_knowledge,
            tick_controls=red_tick_controls,
        )

        blue_knowledge = PlayerKnowledge(
            team=Team.BLUE,
            grid_width=grid_width,
            grid_height=grid_height,
            tick=state.tick,
        )
        blue_knowledge.interrupt_library = initialize_interrupt_library()
        blue_view = PlayerView(
            knowledge=blue_knowledge,
            tick_controls=blue_tick_controls,
        )

        # Create LocalClient
        knowledge_dict = {
            Team.RED: red_view.knowledge,
            Team.BLUE: blue_view.knowledge,
        }
        client = LocalClient(state=state, knowledge=knowledge_dict)

        # Start server if in server mode
        if args.mode == "server":
            from server import GameServer

            server = GameServer(state, knowledge_dict, port=args.port)
            server.start()
            print(f"Server started on port {args.port}")

    views = {
        Team.RED: red_view,
        Team.BLUE: blue_view,
    }

    library_buttons = {
        Team.RED: red_library_btn,
        Team.BLUE: blue_library_btn,
    }

    return GameContext(
        client=client,
        grid_width=grid_width,
        grid_height=grid_height,
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
        library_buttons=library_buttons,
    )


def handle_events(ctx: GameContext) -> bool:
    """Process pygame events. Returns True if game should continue running."""
    for event in pygame.event.get():
        # Process pygame_gui events
        if ctx.ui_manager.process_events(event):
            continue

        if event.type == pygame.QUIT:
            return False

        # Handle window close events
        if event.type == pygame_gui.UI_WINDOW_CLOSE:
            for team in Team:
                view = ctx.views[team]
                if view.library_controls is not None and event.ui_element == view.library_controls.window:
                    view.library_controls = None
                if view.editor_controls is not None and event.ui_element == view.editor_controls.window:
                    view.editor_controls = None

        # Handle interrupt checkbox changes
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            for team in Team:
                view = ctx.views[team]
                if view.plan_controls is not None and view.working_plan is not None:
                    knowledge = ctx.client.get_player_knowledge(team, ctx.client.get_current_tick())
                    # Check if the event is from one of our interrupt checkboxes
                    for int_name, checkbox in view.plan_controls.interrupt_checkboxes.items():
                        if event.ui_element == checkbox:
                            # Toggle this interrupt in the working plan based on checkbox state
                            interrupt = knowledge.interrupt_library.get(int_name)
                            if interrupt:
                                if checkbox.is_checked:
                                    # Add interrupt if not already present
                                    if interrupt not in view.working_plan.interrupts:
                                        view.working_plan.interrupts.append(interrupt)
                                else:
                                    # Remove interrupt if present
                                    if interrupt in view.working_plan.interrupts:
                                        view.working_plan.interrupts.remove(interrupt)
                            break

        for team in Team:
            view = ctx.views[team]

            # Handle pygame_gui button clicks
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                # Handle library button
                if event.ui_element == ctx.library_buttons[team]:
                    if view.library_controls is None:
                        view.library_controls = create_interrupt_library_window(ctx, team)

                # Handle library control buttons
                if view.library_controls is not None:
                    if event.ui_element == view.library_controls.close_btn:
                        view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                        view.library_controls = None
                    elif event.ui_element == view.library_controls.new_btn:
                        # Create a new interrupt with default values
                        new_name = f"new_interrupt_{len(view.knowledge.interrupt_library)}"
                        view.knowledge.interrupt_library[new_name] = Interrupt(
                            condition=IdleCondition(),
                            actions=[MoveHomeAction()],
                        )
                        # Recreate library window to show new interrupt
                        view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                        view.library_controls = create_interrupt_library_window(ctx, team)
                    else:
                        # Check if any per-interrupt button was clicked
                        for int_name, (edit_btn, dup_btn, ren_btn, del_btn) in view.library_controls.interrupt_rows.items():
                            if event.ui_element == edit_btn:
                                if view.editor_controls is None:
                                    view.editor_controls = create_simple_interrupt_editor(ctx, team, int_name)
                                break
                            elif event.ui_element == del_btn:
                                del view.knowledge.interrupt_library[int_name]
                                # Recreate library window
                                view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                                view.library_controls = create_interrupt_library_window(ctx, team)
                                break
                            elif event.ui_element == ren_btn:
                                new_name = f"{int_name}_renamed"
                                interrupt = view.knowledge.interrupt_library[int_name]
                                del view.knowledge.interrupt_library[int_name]
                                view.knowledge.interrupt_library[new_name] = interrupt
                                # Recreate library window
                                view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                                view.library_controls = create_interrupt_library_window(ctx, team)
                                break
                            elif event.ui_element == dup_btn:
                                interrupt = view.knowledge.interrupt_library[int_name]
                                new_name = f"{int_name}_copy"
                                view.knowledge.interrupt_library[new_name] = interrupt
                                # Recreate library window
                                view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                                view.library_controls = create_interrupt_library_window(ctx, team)
                                break

                # Handle editor control buttons
                if view.editor_controls is not None:
                    if event.ui_element == view.editor_controls.cancel_btn:
                        view.editor_controls.window.kill()  # type: ignore[no-untyped-call]
                        view.editor_controls = None
                    elif event.ui_element == view.editor_controls.save_btn:
                        # Save the edited interrupt
                        old_name = view.editor_controls.original_interrupt_name
                        new_name = view.editor_controls.name_entry.get_text()

                        # Build condition based on selected type and parameters
                        condition: Condition[Any] | None = None
                        if view.editor_controls.selected_condition_type:
                            ct = view.editor_controls.selected_condition_type
                            if ct.param_specs:
                                # Read parameters from entries
                                params: dict[str, Any] = {}
                                for param_name, param_type in ct.param_specs.items():
                                    if param_name in view.editor_controls.condition_param_entries:
                                        value_str = view.editor_controls.condition_param_entries[param_name].get_text()
                                        if param_type == int:
                                            params[param_name] = int(value_str) if value_str.isdigit() else 2
                                        elif param_type == Pos:
                                            parts = value_str.split(',')
                                            if len(parts) == 2:
                                                params[param_name] = Pos(int(parts[0]), int(parts[1]))
                                            else:
                                                params[param_name] = Pos(0, 0)
                                condition = ct.factory(**params)
                            else:
                                condition = ct.factory()

                        # Build actions
                        actions = [at.factory() for at, _ in view.editor_controls.selected_actions]

                        if condition and actions:
                            new_interrupt = Interrupt(condition=condition, actions=actions)

                            # Remove old entry if name changed
                            if old_name != new_name and old_name in view.knowledge.interrupt_library:
                                del view.knowledge.interrupt_library[old_name]

                            view.knowledge.interrupt_library[new_name] = new_interrupt

                            # Refresh library window if open
                            if view.library_controls is not None:
                                view.library_controls.window.kill()  # type: ignore[no-untyped-call]
                                view.library_controls = create_interrupt_library_window(ctx, team)

                        view.editor_controls.window.kill()  # type: ignore[no-untyped-call]
                        view.editor_controls = None
                    elif event.ui_element == view.editor_controls.add_action_btn:
                        # Add the selected action from dropdown
                        if view.editor_controls.selected_condition_type:
                            selected_action_name = view.editor_controls.add_action_dropdown.selected_option[0]
                            compatible = get_compatible_actions(view.editor_controls.selected_condition_type.return_type)
                            for at in compatible:
                                if at.name == selected_action_name:
                                    view.editor_controls.selected_actions.append((at, {}))
                                    break
                    elif event.ui_element == view.editor_controls.remove_action_btn:
                        # Remove last action
                        if view.editor_controls.selected_actions:
                            view.editor_controls.selected_actions.pop()

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
                            ctx.client.set_unit_plan(
                                team, view.selected_unit_id, view.working_plan
                            )
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
                                interrupts=get_default_plan_interrupts(view.knowledge.interrupt_library)
                            )
                        break

            # Handle slider value changes
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == view.tick_controls.slider:
                    current_tick = ctx.client.get_current_tick()
                    if current_tick > 0:
                        view.freeze_frame = int(event.value * current_tick)

            # Handle map clicks for unit selection and waypoints
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Skip if click is on a pygame_gui element
                if ctx.ui_manager.get_hovering_any_element():
                    continue

                # Check each team's player view for unit selection or target
                grid_pos = screen_to_grid(
                    mx,
                    my,
                    ctx.team_offsets[team],
                    ctx.views_offset_y,
                    ctx.grid_width,
                    ctx.grid_height,
                )
                # Only allow interaction when viewing live (not a freeze frame)
                if grid_pos is not None and view.freeze_frame is None:
                    # Detect double-click (within 300ms at same position)
                    current_time = pygame.time.get_ticks()
                    is_double_click = (
                        view.last_click_pos == grid_pos
                        and current_time - view.last_click_time < 300
                    )

                    # Update last click tracking
                    view.last_click_time = current_time
                    view.last_click_pos = grid_pos

                    if is_double_click and view.selected_unit_id is not None:
                        # Double-click while unit is selected: issue the working plan
                        if (
                            view.working_plan is not None
                            and view.working_plan.orders
                        ):
                            ctx.client.set_unit_plan(
                                team, view.selected_unit_id, view.working_plan
                            )
                            view.selected_unit_id = None
                            view.working_plan = None
                            # Clean up plan controls
                            if view.plan_controls is not None:
                                view.plan_controls.text_box.kill()
                                view.plan_controls.issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                                view.plan_controls.clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                                view.plan_controls.selection_label.hide()  # type: ignore[no-untyped-call]
                                view.plan_controls = None
                    elif view.selected_unit_id is not None:
                        # Single click with unit selected: append Move order to working plan
                        if view.working_plan is None:
                            view.working_plan = Plan(
                                interrupts=get_default_plan_interrupts(view.knowledge.interrupt_library)
                            )
                        view.working_plan.orders.append(Move(target=grid_pos))
                    else:
                        # Try to select a unit
                        unit = find_unit_at_base(ctx.client, grid_pos, team)
                        if unit is not None:
                            view.selected_unit_id = unit.id
                            # Initialize working plan when selecting a unit
                            view.working_plan = Plan(
                                interrupts=get_default_plan_interrupts(view.knowledge.interrupt_library)
                            )

    return True


def draw_ui(ctx: GameContext) -> None:
    """Render all game views and UI elements."""
    ctx.screen.fill((0, 0, 0))

    # Draw game views
    draw_player_view(
        ctx.screen, ctx.views[Team.RED], Team.RED, ctx.red_offset_x, ctx.views_offset_y
    )
    draw_god_view(
        ctx.screen,
        ctx.client.get_god_view(),
        ctx.god_offset_x,
        ctx.views_offset_y,
        ctx.grid_width,
        ctx.grid_height,
    )
    draw_player_view(
        ctx.screen,
        ctx.views[Team.BLUE],
        Team.BLUE,
        ctx.blue_offset_x,
        ctx.views_offset_y,
    )

    # Update slider positions and labels to reflect current state
    current_tick = ctx.client.get_current_tick()
    if current_tick > 0:
        red_view = ctx.views[Team.RED]
        red_view_tick = red_view.freeze_frame
        if red_view_tick is None:
            red_view.tick_controls.slider.set_current_value(1.0)
            red_view.tick_controls.tick_label.set_text(f"t={current_tick}")
        else:
            red_view.tick_controls.slider.set_current_value(
                red_view_tick / current_tick
            )
            red_view.tick_controls.tick_label.set_text(f"t={red_view_tick}")

        blue_view = ctx.views[Team.BLUE]
        blue_view_tick = blue_view.freeze_frame
        if blue_view_tick is None:
            blue_view.tick_controls.slider.set_current_value(1.0)
            blue_view.tick_controls.tick_label.set_text(f"t={current_tick}")
        else:
            blue_view.tick_controls.slider.set_current_value(
                blue_view_tick / current_tick
            )
            blue_view.tick_controls.tick_label.set_text(f"t={blue_view_tick}")

    # Plan area layout
    plan_y = ctx.slider_y + 30
    interrupts_height = 120
    plan_box_height = 180 - interrupts_height
    btn_y = plan_y + interrupts_height + plan_box_height + 5
    selection_label_y = ctx.slider_y

    # Handle plan controls for each team
    for team in Team:
        view = ctx.views[team]
        plan_offset_x = ctx.team_offsets[team]

        if view.selected_unit_id is not None:
            # Get the selected unit from player knowledge
            knowledge = ctx.client.get_player_knowledge(
                team, ctx.client.get_current_tick()
            )
            selected_unit = None
            if view.selected_unit_id in knowledge.last_seen:
                _, selected_unit = knowledge.last_seen[view.selected_unit_id]

            if selected_unit is None:
                # Unit no longer exists in our knowledge, clear selection
                view.selected_unit_id = None
                view.working_plan = None
                if view.plan_controls is not None:
                    view.plan_controls.text_box.kill()
                    view.plan_controls.issue_plan_btn.kill()  # type: ignore[no-untyped-call]
                    view.plan_controls.clear_plan_btn.kill()  # type: ignore[no-untyped-call]
                    view.plan_controls.selection_label.hide()  # type: ignore[no-untyped-call]
                    if view.plan_controls.interrupt_container:
                        view.plan_controls.interrupt_container.kill()  # type: ignore[no-untyped-call]
                    view.plan_controls = None
                continue

            team_name = team.value

            # Create plan controls if they don't exist
            if view.plan_controls is None:
                selection_label = pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect(
                        plan_offset_x, selection_label_y, ctx.map_pixel_size, 20
                    ),
                    text="",
                    manager=ctx.ui_manager,
                )

                # Interrupt selection container
                interrupt_container = pygame_gui.elements.UIScrollingContainer(
                    relative_rect=pygame.Rect(
                        plan_offset_x, plan_y, ctx.map_pixel_size, interrupts_height
                    ),
                    manager=ctx.ui_manager,
                )

                # Create checkboxes for each interrupt in library
                interrupt_checkboxes: dict[str, pygame_gui.elements.UICheckBox] = {}
                interrupt_labels: dict[str, pygame_gui.elements.UILabel] = {}
                y_off = 0
                for int_name in sorted(knowledge.interrupt_library.keys()):
                    # Create checkbox (small square)
                    checkbox = pygame_gui.elements.UICheckBox(
                        relative_rect=pygame.Rect(5, y_off, 25, 25),
                        text="",
                        manager=ctx.ui_manager,
                        container=interrupt_container,
                    )
                    interrupt_checkboxes[int_name] = checkbox

                    # Create label next to checkbox
                    label = pygame_gui.elements.UILabel(
                        relative_rect=pygame.Rect(35, y_off, ctx.map_pixel_size - 45, 25),
                        text=int_name,
                        manager=ctx.ui_manager,
                        container=interrupt_container,
                    )
                    interrupt_labels[int_name] = label
                    y_off += 30

                interrupt_container.set_scrollable_area_dimensions((ctx.map_pixel_size - 20, max(y_off, interrupts_height)))

                text_box = pygame_gui.elements.UITextBox(
                    html_text="",
                    relative_rect=pygame.Rect(
                        plan_offset_x, plan_y + interrupts_height, ctx.map_pixel_size, plan_box_height
                    ),
                    manager=ctx.ui_manager,
                    wrap_to_height=False,
                )
                issue_plan_btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(plan_offset_x, btn_y, 80, 20),
                    text="Issue Plan",
                    manager=ctx.ui_manager,
                )
                clear_plan_btn = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(plan_offset_x + 90, btn_y, 60, 20),
                    text="Clear",
                    manager=ctx.ui_manager,
                )
                view.plan_controls = PlanControls(
                    text_box=text_box,
                    last_plan_html="",
                    issue_plan_btn=issue_plan_btn,
                    clear_plan_btn=clear_plan_btn,
                    selection_label=selection_label,
                    interrupt_checkboxes=interrupt_checkboxes,
                    interrupt_labels=interrupt_labels,
                    interrupt_container=interrupt_container,
                )

            # Update selection label
            view.plan_controls.selection_label.set_text(
                f"Selected: {team_name} unit - click {team_name}'s map to add waypoints"
            )
            view.plan_controls.selection_label.show()  # type: ignore[no-untyped-call]

            # Update interrupt selections based on working plan
            if view.working_plan is not None:
                active_interrupt_names = set()
                for interrupt in view.working_plan.interrupts:
                    # Find the name of this interrupt in the library
                    for name, lib_interrupt in knowledge.interrupt_library.items():
                        if lib_interrupt is interrupt:
                            active_interrupt_names.add(name)
                            break

                # Update checkbox states
                for int_name, checkbox in view.plan_controls.interrupt_checkboxes.items():
                    should_be_checked = int_name in active_interrupt_names

                    if checkbox.is_checked != should_be_checked:
                        checkbox.set_state(should_be_checked)

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

        # Update interrupt editor window if open
        if view.editor_controls is not None:
            # Update action list display
            action_lines = []
            for i, (at, _) in enumerate(view.editor_controls.selected_actions):
                action_lines.append(f"{i+1}. {at.name}")
            action_html = "<br>".join(action_lines) if action_lines else "<i>No actions</i>"
            view.editor_controls.action_list.set_text(action_html)
            view.editor_controls.action_list.rebuild()

    # Update UI manager
    ctx.ui_manager.update(ctx.clock.tick(60) / 1000.0)

    # Draw UI manager
    ctx.ui_manager.draw_ui(ctx.screen)

    pygame.display.flip()


def main() -> None:
    """Main game loop: initialize, then loop handling events, ticking, and drawing."""
    ctx = initialize_game()

    while True:
        # Handle events
        if not handle_events(ctx):
            break

        # Tick game if appropriate (only in local/server mode)
        if isinstance(ctx.client, LocalClient):
            current_time = pygame.time.get_ticks()
            if current_time - ctx.last_tick >= ctx.tick_interval:
                tick_game(ctx.client.state)
                for team, knowledge in ctx.client.knowledge.items():
                    knowledge.tick_knowledge(ctx.client.state)
                ctx.last_tick = current_time
        elif isinstance(ctx.client, RemoteClient):
            current_time = pygame.time.get_ticks()
            if current_time - ctx.last_tick >= ctx.tick_interval:
                ctx.views[ctx.client.team].knowledge = ctx.client.get_player_knowledge(
                    ctx.client.team, ctx.client.get_current_tick() + 1
                )
                ctx.last_tick = current_time
        else:
            raise ValueError(f"Unknown client type: {type(ctx.client)}")

        # Draw
        draw_ui(ctx)

    pygame.quit()


if __name__ == "__main__":
    main()
