"""Arcade rendering and UI."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Any
import argparse
import numpy
import arcade
import arcade.gui
import time

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
    UnitType,
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

    slider: arcade.gui.UISlider
    tick_label: arcade.gui.UILabel
    live_btn: arcade.gui.UIFlatButton


@dataclass
class DispositionControls:
    """UI controls for unit disposition."""

    fighter_btn: arcade.gui.UIFlatButton
    scout_btn: arcade.gui.UIFlatButton


@dataclass
class PlanControls:
    """UI controls for plan editing."""

    text_area: arcade.gui.UITextArea | None
    last_plan_text: str
    issue_plan_btn: arcade.gui.UIFlatButton | None
    clear_plan_btn: arcade.gui.UIFlatButton | None
    selection_label: arcade.gui.UILabel | None
    fight_checkbox: arcade.gui.UIFlatButton | None
    flee_checkbox: arcade.gui.UIFlatButton | None
    forage_checkbox: arcade.gui.UIFlatButton | None
    come_back_checkbox: arcade.gui.UIFlatButton | None
    checkbox_states: dict[str, bool] = field(default_factory=dict)


@dataclass
class PlayerView:
    knowledge: PlayerKnowledge
    tick_controls: TickControls | None = None
    disposition_controls: DispositionControls | None = None
    plan_controls: PlanControls | None = None
    freeze_frame: Timestamp | None = None
    selected_unit_id: UnitId | None = None
    working_plan: Plan | None = None
    last_click_time: float = 0.0
    last_click_pos: Pos | None = None


@dataclass
class GameContext:
    """Container for all game state and UI elements."""

    client: GameClient
    grid_width: int
    grid_height: int
    views: dict[Team, PlayerView]
    team_offsets: dict[Team, int]
    red_offset_x: int
    god_offset_x: int
    blue_offset_x: int
    views_offset_y: int
    slider_y: int
    map_pixel_size: int
    tick_interval: int
    last_tick: float
    start_time: float  # For get_ticks replacement
    ui_manager: arcade.gui.UIManager | None = None


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
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
    window_height: int,
) -> None:
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    for x in range(grid_width + 1):
        x_pos = offset_x + x * TILE_SIZE
        y_start = window_height - offset_y
        y_end = window_height - (offset_y + map_pixel_height)
        arcade.draw_line(x_pos, y_start, x_pos, y_end, (50, 50, 50))
    for y in range(grid_height + 1):
        y_pos = window_height - (offset_y + y * TILE_SIZE)
        x_start = offset_x
        x_end = offset_x + map_pixel_width
        arcade.draw_line(x_start, y_pos, x_end, y_pos, (50, 50, 50))


def draw_base_cell(
    pos: Pos,
    team: Team,
    offset_x: int,
    offset_y: int,
    window_height: int,
    outline_only: bool = False,
) -> None:
    """Draw a base region with a faint background tint."""
    # Faint background color for base cells
    tint_color = (80, 40, 40) if team == Team.RED else (40, 40, 200)
    left = offset_x + pos.x * TILE_SIZE
    bottom = window_height - (offset_y + pos.y * TILE_SIZE + TILE_SIZE)
    arcade.draw_lbwh_rectangle_filled(left, bottom, TILE_SIZE, TILE_SIZE, tint_color)


def draw_unit_at(
    team: Team,
    pos: Pos,
    offset_x: int,
    offset_y: int,
    window_height: int,
    selected: bool = False,
    outline_only: bool = False,
    unit_type: UnitType = UnitType.FIGHTER,
) -> None:
    color = (255, 100, 100) if team == Team.RED else (100, 100, 255)
    center_x = offset_x + pos.x * TILE_SIZE + TILE_SIZE // 2
    center_y = window_height - (offset_y + pos.y * TILE_SIZE + TILE_SIZE // 2)

    if unit_type == UnitType.SCOUT:
        # Draw square for scouts
        size = TILE_SIZE // 3
        left = center_x - size
        bottom = center_y - size
        width = size * 2
        height = size * 2

        if outline_only:
            arcade.draw_lbwh_rectangle_outline(left, bottom, width, height, color, 1)
        else:
            arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, color)
        if selected:
            arcade.draw_lbwh_rectangle_outline(left - 3, bottom - 3, width + 6, height + 6, (255, 255, 0), 2)
    else:
        # Draw circle for fighters
        radius = TILE_SIZE // 3
        if outline_only:
            arcade.draw_circle_outline(center_x, center_y, radius, color, 1)
        else:
            arcade.draw_circle_filled(center_x, center_y, radius, color)
        if selected:
            arcade.draw_circle_outline(center_x, center_y, radius + 3, (255, 255, 0), 2)


def draw_food(
    food: dict[Pos, int],
    offset_x: int,
    offset_y: int,
    window_height: int,
    outline_only: bool = False,
) -> None:
    """Draw food as small green dots, with multiple items positioned non-overlapping."""
    radius = 3

    for pos, count in food.items():
        tile_x = offset_x + pos.x * TILE_SIZE
        tile_y = window_height - (offset_y + pos.y * TILE_SIZE + TILE_SIZE)

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
                arcade.draw_circle_outline(
                    tile_x + dx, tile_y + dy, radius, (100, 255, 100), 1
                )
            else:
                arcade.draw_circle_filled(
                    tile_x + dx, tile_y + dy, radius, (100, 255, 100)
                )


def draw_god_view(
    state: GameState | None,
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
    window_height: int,
) -> None:
    """Draw the god's-eye view showing the complete game state.

    If state is None (client mode), draws a placeholder message instead.
    """
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    left = offset_x
    bottom = window_height - (offset_y + map_pixel_height)
    arcade.draw_lbwh_rectangle_filled(
        left, bottom, map_pixel_width, map_pixel_height, (30, 30, 30)
    )

    if state is None:
        # Client mode - no god view available
        # Draw a message indicating god view is not available
        text_x = offset_x + map_pixel_width // 2
        text_y = window_height - (offset_y + map_pixel_height // 2)
        arcade.draw_text(
            "God view not available in client mode",
            text_x,
            text_y,
            (100, 100, 100),
            font_size=12,
            anchor_x="center",
            anchor_y="center",
        )
        return

    # Draw grid
    draw_grid(offset_x, offset_y, state.grid_width, state.grid_height, window_height)

    # Draw base regions
    for pos in state.get_base_region(Team.RED).cells:
        draw_base_cell(pos, Team.RED, offset_x, offset_y, window_height)
    for pos in state.get_base_region(Team.BLUE).cells:
        draw_base_cell(pos, Team.BLUE, offset_x, offset_y, window_height)

    # Draw food
    draw_food(state.food, offset_x, offset_y, window_height)

    # Draw units
    for unit in state.units.values():
        draw_unit_at(
            unit.team,
            unit.pos,
            offset_x,
            offset_y,
            window_height,
            unit_type=unit.unit_type,
        )


CELL_BRIGHTNESS_HALFLIFE = 100


def draw_player_view(
    view: PlayerView,
    team: Team,
    offset_x: int,
    offset_y: int,
    window_height: int,
) -> None:
    """Draw a player's view of the map at their selected tick."""
    freeze_frame = view.freeze_frame
    # Use current tick when live (freeze_frame is None), otherwise use freeze_frame
    view_t = view.knowledge.tick if freeze_frame is None else freeze_frame
    logbook = view.knowledge.all_observations

    map_pixel_width = view.knowledge.grid_width * TILE_SIZE
    map_pixel_height = view.knowledge.grid_height * TILE_SIZE
    left = offset_x
    bottom = window_height - (offset_y + map_pixel_height)
    arcade.draw_lbwh_rectangle_filled(
        left, bottom, map_pixel_width, map_pixel_height, (0, 0, 0)
    )

    # Get observations for this timestamp
    cur_observations = logbook.get(view_t, {})

    # Draw cells with gradient tinting based on observation age
    for pos, (last_observed_tick, _) in view.knowledge.last_observations.items():
        age = view_t - last_observed_tick
        # Exponential decay: 80 * 2^-(age/50)
        brightness = int(80 * (2 ** (-(age / CELL_BRIGHTNESS_HALFLIFE))))
        brightness = max(0, min(80, brightness))  # Clamp to [0, 80]

        cell_left = offset_x + pos.x * TILE_SIZE
        cell_bottom = window_height - (offset_y + pos.y * TILE_SIZE + TILE_SIZE)
        arcade.draw_lbwh_rectangle_filled(
            cell_left,
            cell_bottom,
            TILE_SIZE,
            TILE_SIZE,
            (brightness, brightness, brightness),
        )

    # Redraw grid on top
    draw_grid(
        offset_x,
        offset_y,
        view.knowledge.grid_width,
        view.knowledge.grid_height,
        window_height,
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
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                        window_height,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(
                        contents.team,
                        pos,
                        offset_x,
                        offset_y,
                        window_height,
                        outline_only=pos not in cur_observations,
                        unit_type=contents.unit_type,
                    )
                elif isinstance(contents, FoodPresent):
                    draw_food(
                        {pos: contents.count},
                        offset_x,
                        offset_y,
                        window_height,
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
                        pos,
                        contents.team,
                        offset_x,
                        offset_y,
                        window_height,
                    )
                elif isinstance(contents, UnitPresent):
                    draw_unit_at(
                        contents.team,
                        pos,
                        offset_x,
                        offset_y,
                        window_height,
                        outline_only=False,
                        unit_type=contents.unit_type,
                    )
                elif isinstance(contents, FoodPresent):
                    draw_food(
                        {pos: contents.count},
                        offset_x,
                        offset_y,
                        window_height,
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
            # Get unit type from last_seen
            unit_type = UnitType.FIGHTER
            if trajectory.unit_id in view.knowledge.last_seen:
                _, unit = view.knowledge.last_seen[trajectory.unit_id]
                unit_type = unit.unit_type
            draw_unit_at(
                team, predicted_pos, offset_x, offset_y, window_height, outline_only=True, unit_type=unit_type
            )


def screen_to_grid(
    mouse_x: int,
    mouse_y: int,
    offset_x: int,
    offset_y: int,
    grid_width: int,
    grid_height: int,
    window_height: int,
) -> Pos | None:
    # Convert from Arcade coordinates (bottom-left origin) to grid coordinates
    pygame_y = window_height - mouse_y
    rel_x = mouse_x - offset_x
    rel_y = pygame_y - offset_y

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


def make_interrupts_from_checkboxes(
    fight: bool, flee: bool, forage: bool, come_back: bool
) -> list[Interrupt[Any]]:
    """Build list of interrupts based on checkbox states."""
    interrupts: list[Interrupt[Any]] = []
    if fight:
        interrupts.append(attack_nearby_enemy_interrupt)
    if flee:
        interrupts.append(flee_enemy_interrupt)
    if forage:
        interrupts.append(get_food_interrupt)
    if come_back:
        interrupts.append(go_home_when_done_interrupt)
    return interrupts


def make_initial_working_plan_interrupts(unit_type: UnitType) -> list[Interrupt[Any]]:
    """Make the initial interrupts for a working plan based on unit type.

    Fighters get: fight, forage, come home
    Scouts get: flee, forage, come home
    """
    if unit_type == UnitType.SCOUT:
        return make_interrupts_from_checkboxes(
            fight=False, flee=True, forage=True, come_back=True
        )
    else:  # FIGHTER
        return make_interrupts_from_checkboxes(
            fight=True, flee=False, forage=True, come_back=True
        )


class AntGameWindow(arcade.Window):
    """Main window for the Ant RTS game."""

    def __init__(self, ctx: GameContext, window_width: int, window_height: int):
        super().__init__(window_width, window_height, "Ant RTS")
        self.game_ctx = ctx
        self.start_time = time.time()
        self.game_ctx.start_time = self.start_time

        # Set update rate to 60 FPS
        self.set_update_rate(1/60)

    def get_ticks(self) -> float:
        """Get milliseconds since window creation (pygame.time.get_ticks replacement)."""
        return (time.time() - self.start_time) * 1000

    def on_draw(self) -> None:
        """Render the game."""
        self.clear()

        # Draw game views
        draw_player_view(
            self.game_ctx.views[Team.RED],
            Team.RED,
            self.game_ctx.red_offset_x,
            self.game_ctx.views_offset_y,
            self.height,
        )
        draw_god_view(
            self.game_ctx.client.get_god_view(),
            self.game_ctx.god_offset_x,
            self.game_ctx.views_offset_y,
            self.game_ctx.grid_width,
            self.game_ctx.grid_height,
            self.height,
        )
        draw_player_view(
            self.game_ctx.views[Team.BLUE],
            Team.BLUE,
            self.game_ctx.blue_offset_x,
            self.game_ctx.views_offset_y,
            self.height,
        )

        # Update slider positions and labels to reflect current state
        current_tick = self.game_ctx.client.get_current_tick()
        if current_tick > 0:
            red_view = self.game_ctx.views[Team.RED]
            if red_view.tick_controls:
                red_view_tick = red_view.freeze_frame
                if red_view_tick is None:
                    red_view.tick_controls.slider.value = 1.0
                    red_view.tick_controls.tick_label.text = f"t={current_tick}"
                else:
                    red_view.tick_controls.slider.value = red_view_tick / current_tick
                    red_view.tick_controls.tick_label.text = f"t={red_view_tick}"

            blue_view = self.game_ctx.views[Team.BLUE]
            if blue_view.tick_controls:
                blue_view_tick = blue_view.freeze_frame
                if blue_view_tick is None:
                    blue_view.tick_controls.slider.value = 1.0
                    blue_view.tick_controls.tick_label.text = f"t={current_tick}"
                else:
                    blue_view.tick_controls.slider.value = blue_view_tick / current_tick
                    blue_view.tick_controls.tick_label.text = f"t={blue_view_tick}"

        # Update disposition button appearance based on current setting
        if isinstance(self.game_ctx.client, LocalClient):
            for team in Team:
                view = self.game_ctx.views[team]
                if view.disposition_controls:
                    current_disposition = self.game_ctx.client.state.unit_disposition[team]
                    if current_disposition == UnitType.FIGHTER:
                        view.disposition_controls.fighter_btn.text = "Fighter ✓"
                        view.disposition_controls.scout_btn.text = "Scout"
                    else:
                        view.disposition_controls.fighter_btn.text = "Fighter"
                        view.disposition_controls.scout_btn.text = "Scout ✓"

        # Handle plan controls for each team
        for team in Team:
            view = self.game_ctx.views[team]
            plan_offset_x = self.game_ctx.team_offsets[team]

            # Plan area layout (moved down to avoid overlapping disposition buttons)
            plan_y = self.game_ctx.slider_y + 55
            arcade_plan_y = self.height - plan_y
            plan_box_width = self.game_ctx.map_pixel_size
            plan_box_height = 180
            btn_y_offset = -plan_box_height - 5
            selection_label_y = self.height - self.game_ctx.slider_y

            if view.selected_unit_id is not None:
                # Get the selected unit from player knowledge
                knowledge = self.game_ctx.client.get_player_knowledge(
                    team, self.game_ctx.client.get_current_tick()
                )
                selected_unit = None
                if view.selected_unit_id in knowledge.last_seen:
                    _, selected_unit = knowledge.last_seen[view.selected_unit_id]

                if selected_unit is None:
                    # Unit no longer exists in our knowledge, clear selection
                    view.selected_unit_id = None
                    view.working_plan = None
                    if view.plan_controls is not None:
                        # Clean up plan controls
                        view.plan_controls = None
                    continue

                team_name = team.value

                # Display the working plan
                if view.working_plan is not None:
                    plan_lines = format_plan(view.working_plan, selected_unit)
                    plan_text = "\n".join(plan_lines)

                    # Draw selection label
                    label_text = f"Selected: {team_name} unit - click {team_name}'s map to add waypoints"
                    arcade.draw_text(
                        label_text,
                        plan_offset_x,
                        selection_label_y,
                        arcade.color.WHITE,
                        font_size=10,
                    )

                    # Draw plan text box background
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x,
                        arcade_plan_y - plan_box_height,
                        plan_box_width,
                        plan_box_height,
                        (40, 40, 40),
                    )
                    arcade.draw_lbwh_rectangle_outline(
                        plan_offset_x,
                        arcade_plan_y - plan_box_height,
                        plan_box_width,
                        plan_box_height,
                        arcade.color.WHITE,
                        2
                    )

                    # Draw plan text
                    arcade.draw_text(
                        plan_text,
                        plan_offset_x + 5,
                        arcade_plan_y - 20,
                        arcade.color.WHITE,
                        font_size=10,
                        multiline=True,
                        width=plan_box_width - 10,
                    )

                    # Draw buttons
                    btn_y = arcade_plan_y + btn_y_offset - 5

                    # Issue Plan button
                    has_orders = view.working_plan and view.working_plan.orders
                    issue_btn_color = arcade.color.GREEN if has_orders else arcade.color.GRAY
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x,
                        btn_y,
                        80,
                        20,
                        issue_btn_color,
                    )
                    arcade.draw_text(
                        "Issue Plan",
                        plan_offset_x + 5,
                        btn_y + 5,
                        arcade.color.BLACK,
                        font_size=10,
                    )

                    # Clear button
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x + 90,
                        btn_y,
                        60,
                        20,
                        arcade.color.RED,
                    )
                    arcade.draw_text(
                        "Clear",
                        plan_offset_x + 100,
                        btn_y + 5,
                        arcade.color.BLACK,
                        font_size=10,
                    )

                    # Initialize or update plan controls
                    if view.plan_controls is None:
                        # First time creating plan controls - initialize checkboxes based on unit type
                        if selected_unit.unit_type == UnitType.SCOUT:
                            default_checkboxes = {"fight": False, "flee": True, "forage": True, "come_back": True}
                        else:  # FIGHTER
                            default_checkboxes = {"fight": True, "flee": False, "forage": True, "come_back": True}

                        view.plan_controls = PlanControls(
                            text_area=None,
                            last_plan_text=plan_text,
                            issue_plan_btn=None,
                            clear_plan_btn=None,
                            selection_label=None,
                            fight_checkbox=None,
                            flee_checkbox=None,
                            forage_checkbox=None,
                            come_back_checkbox=None,
                            checkbox_states=default_checkboxes,
                        )
                    else:
                        # Update plan text while preserving checkbox states
                        view.plan_controls.last_plan_text = plan_text

                    # Draw interrupt checkboxes
                    checkbox_y = btn_y - 25
                    checkbox_width = 80
                    checkbox_spacing = 90

                    # Fight checkbox
                    fight_checked = view.plan_controls.checkbox_states.get("fight", True)
                    fight_color = (60, 100, 60) if fight_checked else (60, 60, 60)
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x,
                        checkbox_y,
                        checkbox_width,
                        20,
                        fight_color,
                    )
                    arcade.draw_text(
                        f"{"☑" if fight_checked else "☐"} fight",
                        plan_offset_x + 5,
                        checkbox_y + 5,
                        arcade.color.WHITE,
                        font_size=10,
                    )

                    # Flee checkbox
                    flee_checked = view.plan_controls.checkbox_states.get("flee", True)
                    flee_color = (60, 100, 60) if flee_checked else (60, 60, 60)
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x + checkbox_spacing,
                        checkbox_y,
                        checkbox_width,
                        20,
                        flee_color,
                    )
                    arcade.draw_text(
                        f"{"☑" if flee_checked else "☐"} flee",
                        plan_offset_x + checkbox_spacing + 5,
                        checkbox_y + 5,
                        arcade.color.WHITE,
                        font_size=10,
                    )

                    # Forage checkbox
                    forage_checked = view.plan_controls.checkbox_states.get("forage", True)
                    forage_color = (60, 100, 60) if forage_checked else (60, 60, 60)
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x + checkbox_spacing * 2,
                        checkbox_y,
                        checkbox_width,
                        20,
                        forage_color,
                    )
                    arcade.draw_text(
                        f"{"☑" if forage_checked else "☐"} forage",
                        plan_offset_x + checkbox_spacing * 2 + 5,
                        checkbox_y + 5,
                        arcade.color.WHITE,
                        font_size=10,
                    )

                    # Come back checkbox
                    come_back_checked = view.plan_controls.checkbox_states.get("come_back", True)
                    come_back_color = (60, 100, 60) if come_back_checked else (60, 60, 60)
                    arcade.draw_lbwh_rectangle_filled(
                        plan_offset_x + checkbox_spacing * 3,
                        checkbox_y,
                        checkbox_width,
                        20,
                        come_back_color,
                    )
                    arcade.draw_text(
                        f"{"☑" if come_back_checked else "☐"} come back",
                        plan_offset_x + checkbox_spacing * 3 + 5,
                        checkbox_y + 5,
                        arcade.color.WHITE,
                        font_size=10,
                    )

            else:
                # No selected unit for this team, clean up plan controls if they exist
                if view.plan_controls is not None:
                    view.plan_controls = None

        # Draw UI manager
        if self.game_ctx.ui_manager:
            self.game_ctx.ui_manager.draw()

    def on_update(self, delta_time: float) -> None:
        """Update game state."""
        # Tick game if appropriate (only in local/server mode)
        if isinstance(self.game_ctx.client, LocalClient):
            current_time = self.get_ticks()
            if current_time - self.game_ctx.last_tick >= self.game_ctx.tick_interval:
                tick_game(self.game_ctx.client.state)
                for team, knowledge in self.game_ctx.client.knowledge.items():
                    knowledge.tick_knowledge(self.game_ctx.client.state)
                self.game_ctx.last_tick = current_time
        elif isinstance(self.game_ctx.client, RemoteClient):
            current_time = self.get_ticks()
            if current_time - self.game_ctx.last_tick >= self.game_ctx.tick_interval:
                self.game_ctx.views[self.game_ctx.client.team].knowledge = self.game_ctx.client.get_player_knowledge(
                    self.game_ctx.client.team, self.game_ctx.client.get_current_tick() + 1
                )
                self.game_ctx.last_tick = current_time
        else:
            raise ValueError(f"Unknown client type: {type(self.game_ctx.client)}")

        # Update UI manager
        if self.game_ctx.ui_manager:
            self.game_ctx.ui_manager.on_update(delta_time)  # type: ignore[no-untyped-call]

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        """Handle mouse clicks."""
        # Let UI manager handle it first
        if self.game_ctx.ui_manager and self.game_ctx.ui_manager.on_mouse_press(x, y, button, modifiers):
            return

        # Only handle left clicks
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        # Check for plan control button clicks first
        for team in Team:
            view = self.game_ctx.views[team]
            if view.selected_unit_id is not None and view.working_plan is not None:
                plan_offset_x = self.game_ctx.team_offsets[team]
                plan_y = self.game_ctx.slider_y + 30
                arcade_plan_y = self.height - plan_y
                plan_box_height = 180
                btn_y_offset = -plan_box_height - 5
                btn_y = arcade_plan_y + btn_y_offset - 5

                # Check Issue Plan button (80x20 at plan_offset_x, btn_y)
                if (plan_offset_x <= x <= plan_offset_x + 80 and
                    btn_y <= y <= btn_y + 20):
                    # Issue plan button clicked
                    if view.working_plan.orders:
                        self.game_ctx.client.set_unit_plan(
                            team, view.selected_unit_id, view.working_plan
                        )
                        view.selected_unit_id = None
                        view.working_plan = None
                        view.plan_controls = None
                    return

                # Check Clear button (60x20 at plan_offset_x + 90, btn_y)
                if (plan_offset_x + 90 <= x <= plan_offset_x + 150 and
                    btn_y <= y <= btn_y + 20):
                    # Clear button clicked - get unit type for proper defaults
                    knowledge = self.game_ctx.client.get_player_knowledge(
                        team, self.game_ctx.client.get_current_tick()
                    )
                    unit_type = UnitType.FIGHTER  # default
                    if view.selected_unit_id in knowledge.last_seen:
                        _, selected_unit = knowledge.last_seen[view.selected_unit_id]
                        unit_type = selected_unit.unit_type

                    view.working_plan = Plan(interrupts=make_initial_working_plan_interrupts(unit_type))
                    # Reset checkboxes to default states based on unit type
                    if view.plan_controls:
                        if unit_type == UnitType.SCOUT:
                            view.plan_controls.checkbox_states = {
                                "fight": False, "flee": True, "forage": True, "come_back": True
                            }
                        else:  # FIGHTER
                            view.plan_controls.checkbox_states = {
                                "fight": True, "flee": False, "forage": True, "come_back": True
                            }
                    return

                # Check checkbox clicks (80x20 each, positioned below buttons)
                checkbox_y = btn_y - 25
                checkbox_width = 80
                checkbox_spacing = 90

                # Fight checkbox
                if (plan_offset_x <= x <= plan_offset_x + checkbox_width and
                    checkbox_y <= y <= checkbox_y + 20):
                    if view.plan_controls and view.working_plan:
                        current = view.plan_controls.checkbox_states.get("fight", True)
                        view.plan_controls.checkbox_states["fight"] = not current
                        # Update working plan interrupts
                        view.working_plan.interrupts = make_interrupts_from_checkboxes(
                            fight=view.plan_controls.checkbox_states["fight"],
                            flee=view.plan_controls.checkbox_states["flee"],
                            forage=view.plan_controls.checkbox_states["forage"],
                            come_back=view.plan_controls.checkbox_states["come_back"],
                        )
                    return

                # Flee checkbox
                if (plan_offset_x + checkbox_spacing <= x <= plan_offset_x + checkbox_spacing + checkbox_width and
                    checkbox_y <= y <= checkbox_y + 20):
                    if view.plan_controls and view.working_plan:
                        current = view.plan_controls.checkbox_states.get("flee", True)
                        view.plan_controls.checkbox_states["flee"] = not current
                        # Update working plan interrupts
                        view.working_plan.interrupts = make_interrupts_from_checkboxes(
                            fight=view.plan_controls.checkbox_states["fight"],
                            flee=view.plan_controls.checkbox_states["flee"],
                            forage=view.plan_controls.checkbox_states["forage"],
                            come_back=view.plan_controls.checkbox_states["come_back"],
                        )
                    return

                # Forage checkbox
                if (plan_offset_x + checkbox_spacing * 2 <= x <= plan_offset_x + checkbox_spacing * 2 + checkbox_width and
                    checkbox_y <= y <= checkbox_y + 20):
                    if view.plan_controls and view.working_plan:
                        current = view.plan_controls.checkbox_states.get("forage", True)
                        view.plan_controls.checkbox_states["forage"] = not current
                        # Update working plan interrupts
                        view.working_plan.interrupts = make_interrupts_from_checkboxes(
                            fight=view.plan_controls.checkbox_states["fight"],
                            flee=view.plan_controls.checkbox_states["flee"],
                            forage=view.plan_controls.checkbox_states["forage"],
                            come_back=view.plan_controls.checkbox_states["come_back"],
                        )
                    return

                # Come back checkbox
                if (plan_offset_x + checkbox_spacing * 3 <= x <= plan_offset_x + checkbox_spacing * 3 + checkbox_width and
                    checkbox_y <= y <= checkbox_y + 20):
                    if view.plan_controls and view.working_plan:
                        current = view.plan_controls.checkbox_states.get("come_back", True)
                        view.plan_controls.checkbox_states["come_back"] = not current
                        # Update working plan interrupts
                        view.working_plan.interrupts = make_interrupts_from_checkboxes(
                            fight=view.plan_controls.checkbox_states["fight"],
                            flee=view.plan_controls.checkbox_states["flee"],
                            forage=view.plan_controls.checkbox_states["forage"],
                            come_back=view.plan_controls.checkbox_states["come_back"],
                        )
                    return

        for team in Team:
            view = self.game_ctx.views[team]

            # Check each team's player view for unit selection or target
            grid_pos = screen_to_grid(
                x,
                y,
                self.game_ctx.team_offsets[team],
                self.game_ctx.views_offset_y,
                self.game_ctx.grid_width,
                self.game_ctx.grid_height,
                self.height,
            )
            # Only allow interaction when viewing live (not a freeze frame)
            if grid_pos is not None and view.freeze_frame is None:
                # Detect double-click (within 300ms at same position)
                current_time = self.get_ticks()
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
                        self.game_ctx.client.set_unit_plan(
                            team, view.selected_unit_id, view.working_plan
                        )
                        view.selected_unit_id = None
                        view.working_plan = None
                        # Clean up plan controls
                        if view.plan_controls is not None:
                            view.plan_controls = None
                elif view.selected_unit_id is not None:
                    # Single click with unit selected: append Move order to working plan
                    if view.working_plan is None:
                        # Look up the selected unit to get its type
                        knowledge = self.game_ctx.client.get_player_knowledge(
                            team, self.game_ctx.client.get_current_tick()
                        )
                        unit_type = UnitType.FIGHTER  # default
                        if view.selected_unit_id in knowledge.last_seen:
                            _, selected_unit = knowledge.last_seen[view.selected_unit_id]
                            unit_type = selected_unit.unit_type
                        view.working_plan = Plan(
                            interrupts=make_initial_working_plan_interrupts(unit_type)
                        )
                    view.working_plan.orders.append(Move(target=grid_pos))
                else:
                    # Try to select a unit
                    unit = find_unit_at_base(self.game_ctx.client, grid_pos, team)
                    if unit is not None:
                        view.selected_unit_id = unit.id
                        # Initialize working plan when selecting a unit
                        view.working_plan = Plan(
                            interrupts=make_initial_working_plan_interrupts(unit.unit_type)
                        )


def initialize_game() -> tuple[GameContext, AntGameWindow]:
    """Parse arguments and initialize game state and UI elements."""
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

    tick_interval = 200
    last_tick = 0.0

    # Create views and client based on mode
    client: GameClient
    if args.mode == "client":
        # Client mode: only create view for the client's team
        # Reuse the temp_client we created earlier
        client = temp_client
        client_team = Team[args.team]

        # Create a placeholder knowledge that will be updated by fetches
        if client_team == Team.RED:
            red_view = PlayerView(
                knowledge=initial_knowledge,
            )
            # Create a dummy blue view (won't be visible)
            blue_view = PlayerView(
                knowledge=PlayerKnowledge(
                    team=Team.BLUE,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    tick=Timestamp(0),
                ),
            )
        else:
            # Create a dummy red view (won't be visible)
            red_view = PlayerView(
                knowledge=PlayerKnowledge(
                    team=Team.RED,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    tick=Timestamp(0),
                ),
            )
            blue_view = PlayerView(
                knowledge=initial_knowledge,
            )
    else:
        # Local or server mode: create views for both teams
        assert state is not None, "State should not be None in local/server mode"

        red_view = PlayerView(
            knowledge=PlayerKnowledge(
                team=Team.RED,
                grid_width=grid_width,
                grid_height=grid_height,
                tick=state.tick,
            ),
        )
        blue_view = PlayerView(
            knowledge=PlayerKnowledge(
                team=Team.BLUE,
                grid_width=grid_width,
                grid_height=grid_height,
                tick=state.tick,
            ),
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

    ctx = GameContext(
        client=client,
        grid_width=grid_width,
        grid_height=grid_height,
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
        start_time=0.0,  # Will be set by window
    )

    # Create window first (required for UIManager and widgets)
    window = AntGameWindow(ctx, window_width, window_height)

    # Now create UIManager with the window
    ui_manager = arcade.gui.UIManager()
    ctx.ui_manager = ui_manager
    ui_manager.enable()

    # Now that window exists, create UI widgets
    # Create time sliders for RED team
    red_slider = arcade.gui.UISlider(value=0, min_value=0, max_value=1, width=slider_width, height=20)
    red_tick_label = arcade.gui.UILabel(text="t=0", width=45, height=20)
    red_live_btn = arcade.gui.UIFlatButton(text="LIVE", width=50, height=20)

    red_tick_controls = TickControls(
        slider=red_slider,
        tick_label=red_tick_label,
        live_btn=red_live_btn,
    )

    # Create time sliders for BLUE team
    blue_slider = arcade.gui.UISlider(value=0, min_value=0, max_value=1, width=slider_width, height=20)
    blue_tick_label = arcade.gui.UILabel(text="t=0", width=45, height=20)
    blue_live_btn = arcade.gui.UIFlatButton(text="LIVE", width=50, height=20)

    blue_tick_controls = TickControls(
        slider=blue_slider,
        tick_label=blue_tick_label,
        live_btn=blue_live_btn,
    )

    # Create disposition controls for RED team
    red_fighter_btn = arcade.gui.UIFlatButton(text="Fighter", width=70, height=20)
    red_scout_btn = arcade.gui.UIFlatButton(text="Scout", width=70, height=20)

    red_disposition_controls = DispositionControls(
        fighter_btn=red_fighter_btn,
        scout_btn=red_scout_btn,
    )

    # Create disposition controls for BLUE team
    blue_fighter_btn = arcade.gui.UIFlatButton(text="Fighter", width=70, height=20)
    blue_scout_btn = arcade.gui.UIFlatButton(text="Scout", width=70, height=20)

    blue_disposition_controls = DispositionControls(
        fighter_btn=blue_fighter_btn,
        scout_btn=blue_scout_btn,
    )

    # Assign tick controls and disposition controls to views
    ctx.views[Team.RED].tick_controls = red_tick_controls
    ctx.views[Team.RED].disposition_controls = red_disposition_controls
    ctx.views[Team.BLUE].tick_controls = blue_tick_controls
    ctx.views[Team.BLUE].disposition_controls = blue_disposition_controls

    # Add slider event handlers
    def on_red_slider_change(event: Any) -> None:
        current_tick = ctx.client.get_current_tick()
        if current_tick > 0:
            ctx.views[Team.RED].freeze_frame = int(red_slider.value * current_tick)

    def on_blue_slider_change(event: Any) -> None:
        current_tick = ctx.client.get_current_tick()
        if current_tick > 0:
            ctx.views[Team.BLUE].freeze_frame = int(blue_slider.value * current_tick)

    def on_red_live_click(event: Any) -> None:
        ctx.views[Team.RED].freeze_frame = None

    def on_blue_live_click(event: Any) -> None:
        ctx.views[Team.BLUE].freeze_frame = None

    # Disposition button handlers
    def on_red_fighter_click(event: Any) -> None:
        ctx.client.set_unit_disposition(Team.RED, UnitType.FIGHTER)

    def on_red_scout_click(event: Any) -> None:
        ctx.client.set_unit_disposition(Team.RED, UnitType.SCOUT)

    def on_blue_fighter_click(event: Any) -> None:
        ctx.client.set_unit_disposition(Team.BLUE, UnitType.FIGHTER)

    def on_blue_scout_click(event: Any) -> None:
        ctx.client.set_unit_disposition(Team.BLUE, UnitType.SCOUT)

    # Register event handlers using decorators
    @red_slider.event("on_change")
    def _red_slider_change(event: Any) -> None:
        on_red_slider_change(event)

    @blue_slider.event("on_change")
    def _blue_slider_change(event: Any) -> None:
        on_blue_slider_change(event)

    @red_live_btn.event("on_click")
    def _red_live_click(event: Any) -> None:
        on_red_live_click(event)

    @blue_live_btn.event("on_click")
    def _blue_live_click(event: Any) -> None:
        on_blue_live_click(event)

    @red_fighter_btn.event("on_click")
    def _red_fighter_click(event: Any) -> None:
        on_red_fighter_click(event)

    @red_scout_btn.event("on_click")
    def _red_scout_click(event: Any) -> None:
        on_red_scout_click(event)

    @blue_fighter_btn.event("on_click")
    def _blue_fighter_click(event: Any) -> None:
        on_blue_fighter_click(event)

    @blue_scout_btn.event("on_click")
    def _blue_scout_click(event: Any) -> None:
        on_blue_scout_click(event)

    # Layout widgets using UIAnchorLayout
    # Convert from pygame top-left coordinates to arcade bottom-left
    arcade_slider_y = window_height - slider_y - 20
    disposition_y = slider_y + 25
    arcade_disposition_y = window_height - disposition_y - 20

    # RED team disposition buttons
    red_disposition_box = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
    red_disposition_box.add(red_fighter_btn)
    red_disposition_box.add(red_scout_btn)
    red_disposition_anchor = arcade.gui.UIAnchorLayout()
    red_disposition_anchor.add(
        red_disposition_box,
        anchor_x="left",
        anchor_y="bottom",
        align_x=red_offset_x,
        align_y=arcade_disposition_y
    )
    ui_manager.add(red_disposition_anchor)

    # RED team slider widgets
    red_h_box = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
    red_h_box.add(red_slider)
    red_h_box.add(red_tick_label)
    red_h_box.add(red_live_btn)
    red_anchor = arcade.gui.UIAnchorLayout()
    red_anchor.add(
        red_h_box,
        anchor_x="left",
        anchor_y="bottom",
        align_x=red_offset_x,
        align_y=arcade_slider_y
    )
    ui_manager.add(red_anchor)

    # BLUE team disposition buttons
    blue_disposition_box = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
    blue_disposition_box.add(blue_fighter_btn)
    blue_disposition_box.add(blue_scout_btn)
    blue_disposition_anchor = arcade.gui.UIAnchorLayout()
    blue_disposition_anchor.add(
        blue_disposition_box,
        anchor_x="left",
        anchor_y="bottom",
        align_x=blue_offset_x,
        align_y=arcade_disposition_y
    )
    ui_manager.add(blue_disposition_anchor)

    # BLUE team slider widgets
    blue_h_box = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
    blue_h_box.add(blue_slider)
    blue_h_box.add(blue_tick_label)
    blue_h_box.add(blue_live_btn)
    blue_anchor = arcade.gui.UIAnchorLayout()
    blue_anchor.add(
        blue_h_box,
        anchor_x="left",
        anchor_y="bottom",
        align_x=blue_offset_x,
        align_y=arcade_slider_y
    )
    ui_manager.add(blue_anchor)

    return ctx, window


def main() -> None:
    """Main game loop: initialize, then run the arcade window."""
    ctx, window = initialize_game()
    arcade.run()


if __name__ == "__main__":
    main()
