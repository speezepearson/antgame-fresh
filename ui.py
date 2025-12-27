"""Arcade rendering and UI using Sections for modular layout."""

from __future__ import annotations
from dataclasses import dataclass, field
import random
import time
from typing import Any
import arcade
import arcade.gui
from arcade import Section, SectionManager

from core import Pos, Region, Timestamp
from knowledge import PlayerKnowledge
from client import GameClient, LocalClient, RemoteClient
from game_lifecycle import GameLifecycle, parse_args, create_lifecycle_from_args
from mechanics import (
    CreateUnitPlayerAction,
    Empty,
    FoodPresent,
    GameState,
    Team,
    BasePresent,
    UnitPresent,
    Unit,
    UnitId,
    UnitType,
)
from planning import EnemyInRangeCondition, FoodInRangeCondition, IdleCondition, Interrupt, Move, MoveHomeAction, MoveThereAction, Plan, PlanningMind, ResumeAction, SetUnitPlanPlayerAction


# Rendering Constants
TILE_SIZE = 16
PADDING = 10

# Checkbox textures for UITextureToggle
TEX_CHECKBOX_CHECKED = arcade.load_texture(":resources:gui_basic_assets/checkbox/blue_check.png")
TEX_CHECKBOX_UNCHECKED = arcade.load_texture(":resources:gui_basic_assets/checkbox/empty.png")


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

    container: arcade.gui.UIWidget
    text_area: arcade.gui.UITextArea
    issue_plan_btn: arcade.gui.UIFlatButton
    clear_plan_btn: arcade.gui.UIFlatButton
    selection_label: arcade.gui.UILabel
    fight_checkbox: arcade.gui.UITextureToggle
    flee_checkbox: arcade.gui.UITextureToggle
    forage_checkbox: arcade.gui.UITextureToggle
    come_back_checkbox: arcade.gui.UITextureToggle


@dataclass
class PlayerViewState:
    """Mutable state for a player's view."""
    knowledge: PlayerKnowledge
    freeze_frame: Timestamp | None = None
    selected_unit_id: UnitId | None = None
    working_plan: Plan | None = None
    last_click_time: float = 0.0
    last_click_pos: Pos | None = None


# Interrupt definitions
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
    """Make the initial interrupts for a working plan based on unit type."""
    if unit_type == UnitType.SCOUT:
        return make_interrupts_from_checkboxes(
            fight=False, flee=True, forage=True, come_back=True
        )
    else:
        return make_interrupts_from_checkboxes(
            fight=True, flee=False, forage=True, come_back=True
        )


def format_plan(plan: Plan, unit: Unit) -> list[str]:
    """Format a plan as a list of strings for display."""
    lines = []
    if plan.orders:
        lines.append("Orders:")
        for i, order in enumerate(plan.orders):
            prefix = "> " if i == 0 else "  "
            lines.append(f"{prefix}{order.description}")
    else:
        lines.append("Orders: (none)")

    if plan.interrupts:
        lines.append("Interrupts:")
        for interrupt in plan.interrupts:
            condition_desc = interrupt.condition.description
            lines.append(
                f"  If {condition_desc}: {'; '.join([action.description for action in interrupt.actions])}"
            )
    return lines


# Drawing helpers (use absolute window coordinates with offset)
CELL_BRIGHTNESS_HALFLIFE = 100


def draw_grid(offset_x: int, offset_y: int, width: int, height: int, grid_width: int, grid_height: int) -> None:
    """Draw grid lines at offset position."""
    for x in range(grid_width + 1):
        x_pos = offset_x + x * TILE_SIZE
        arcade.draw_line(x_pos, offset_y, x_pos, offset_y + height, (50, 50, 50))
    for y in range(grid_height + 1):
        y_pos = offset_y + y * TILE_SIZE
        arcade.draw_line(offset_x, y_pos, offset_x + width, y_pos, (50, 50, 50))


def draw_base_cell(pos: Pos, team: Team, offset_x: int, offset_y: int, map_height: int) -> None:
    """Draw a base region cell."""
    tint_color = (80, 40, 40) if team == Team.RED else (40, 40, 200)
    left = offset_x + pos.x * TILE_SIZE
    bottom = offset_y + map_height - (pos.y + 1) * TILE_SIZE
    arcade.draw_lbwh_rectangle_filled(left, bottom, TILE_SIZE, TILE_SIZE, tint_color)


def draw_unit_at(
    team: Team,
    pos: Pos,
    offset_x: int,
    offset_y: int,
    map_height: int,
    selected: bool = False,
    outline_only: bool = False,
    unit_type: UnitType = UnitType.FIGHTER,
) -> None:
    """Draw a unit at grid position."""
    color = (255, 100, 100) if team == Team.RED else (100, 100, 255)
    center_x = offset_x + pos.x * TILE_SIZE + TILE_SIZE // 2
    center_y = offset_y + map_height - (pos.y * TILE_SIZE + TILE_SIZE // 2)

    if unit_type == UnitType.SCOUT:
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
        radius = TILE_SIZE // 3
        if outline_only:
            arcade.draw_circle_outline(center_x, center_y, radius, color, 1)
        else:
            arcade.draw_circle_filled(center_x, center_y, radius, color)
        if selected:
            arcade.draw_circle_outline(center_x, center_y, radius + 3, (255, 255, 0), 2)


def draw_food(food: dict[Pos, int], offset_x: int, offset_y: int, map_height: int, outline_only: bool = False) -> None:
    """Draw food as small green dots."""
    radius = 3
    for pos, count in food.items():
        tile_x = offset_x + pos.x * TILE_SIZE
        tile_y = offset_y + map_height - (pos.y + 1) * TILE_SIZE

        if count == 1:
            positions = [(TILE_SIZE // 2, TILE_SIZE // 2)]
        elif count == 2:
            positions = [(TILE_SIZE // 3, TILE_SIZE // 2), (2 * TILE_SIZE // 3, TILE_SIZE // 2)]
        elif count == 3:
            positions = [
                (TILE_SIZE // 2, TILE_SIZE // 3),
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),
            ]
        elif count == 4:
            positions = [
                (TILE_SIZE // 3, TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, TILE_SIZE // 3),
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),
            ]
        else:
            positions = [
                (TILE_SIZE // 3, TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, TILE_SIZE // 3),
                (TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (2 * TILE_SIZE // 3, 2 * TILE_SIZE // 3),
                (TILE_SIZE // 2, TILE_SIZE // 2),
            ][: min(count, 5)]

        for dx, dy in positions:
            if outline_only:
                arcade.draw_circle_outline(tile_x + dx, tile_y + dy, radius, (100, 255, 100), 1)
            else:
                arcade.draw_circle_filled(tile_x + dx, tile_y + dy, radius, (100, 255, 100))


def screen_to_grid(mouse_x: int, mouse_y: int, grid_width: int, grid_height: int, map_height: int) -> Pos | None:
    """Convert section-relative coordinates to grid position."""
    map_pixel_width = grid_width * TILE_SIZE
    map_pixel_height = grid_height * TILE_SIZE
    # mouse_y is in arcade coords (0 at bottom), convert to grid coords (0 at top)
    grid_y = (map_height - mouse_y) // TILE_SIZE
    grid_x = mouse_x // TILE_SIZE
    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
        return Pos(grid_x, grid_y)
    return None


class GodMapSection(Section):
    """Section displaying the God's-eye view of the entire game state."""

    def __init__(
        self,
        left: int,
        bottom: int,
        width: int,
        height: int,
        lifecycle: GameLifecycle,
    ):
        super().__init__(left, bottom, width, height, accept_keyboard_keys=False)
        self.lifecycle = lifecycle
        self.map_height = height

    @property
    def grid_width(self) -> int:
        return self.lifecycle.grid_width

    @property
    def grid_height(self) -> int:
        return self.lifecycle.grid_height

    def on_draw(self) -> None:
        state = self.lifecycle.state
        map_pixel_width = self.grid_width * TILE_SIZE
        map_pixel_height = self.grid_height * TILE_SIZE
        ox, oy = int(self.left), int(self.bottom)

        arcade.draw_lbwh_rectangle_filled(ox, oy, map_pixel_width, map_pixel_height, (30, 30, 30))

        if state is None:
            text_x = ox + map_pixel_width // 2
            text_y = oy + map_pixel_height // 2
            arcade.draw_text(
                "God view not available in client mode",
                text_x, text_y, (100, 100, 100),
                font_size=12, anchor_x="center", anchor_y="center",
            )
            return

        draw_grid(ox, oy, map_pixel_width, map_pixel_height, self.grid_width, self.grid_height)

        for pos in state.get_base_region(Team.RED).cells:
            draw_base_cell(pos, Team.RED, ox, oy, map_pixel_height)
        for pos in state.get_base_region(Team.BLUE).cells:
            draw_base_cell(pos, Team.BLUE, ox, oy, map_pixel_height)

        draw_food(state.food, ox, oy, map_pixel_height)

        for unit in state.units.values():
            draw_unit_at(unit.team, unit.pos, ox, oy, map_pixel_height, unit_type=unit.unit_type)


class PlayerMapSection(Section):
    """Section displaying a single player's view of the map with controls."""

    def __init__(
        self,
        left: int,
        bottom: int,
        width: int,
        height: int,
        team: Team,
        lifecycle: GameLifecycle,
        view_state: PlayerViewState,
        ui_manager: arcade.gui.UIManager,
        start_time_getter: Any,  # callable returning elapsed time
    ):
        super().__init__(left, bottom, width, height, accept_keyboard_keys=False)
        self.team = team
        self.lifecycle = lifecycle
        self.view_state = view_state
        self.ui_manager = ui_manager
        self.start_time_getter = start_time_getter

        self.map_pixel_size = self.lifecycle.grid_width * TILE_SIZE
        self.map_height = self.map_pixel_size

        # UI controls
        self.tick_controls: TickControls | None = None
        self.disposition_controls: DispositionControls | None = None
        self.plan_controls: PlanControls | None = None

        self._setup_controls()

    @property
    def client(self) -> GameClient:
        return self.lifecycle.client

    @property
    def grid_width(self) -> int:
        return self.lifecycle.grid_width

    @property
    def grid_height(self) -> int:
        return self.lifecycle.grid_height

    def _setup_controls(self) -> None:
        """Create tick and disposition controls using UIBoxLayout."""
        # Main vertical layout for all controls below the map
        controls_container = arcade.gui.UIBoxLayout(vertical=True, space_between=5)

        # Tick control row: slider | label | live button
        slider_width = self.map_pixel_size - 100
        slider = arcade.gui.UISlider(value=0, min_value=0, max_value=1, width=slider_width, height=20)
        tick_label = arcade.gui.UILabel(text="t=0", width=45, height=20)
        live_btn = arcade.gui.UIFlatButton(text="LIVE", width=50, height=20)

        tick_row = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
        tick_row.add(slider)
        tick_row.add(tick_label)
        tick_row.add(live_btn)
        controls_container.add(tick_row)

        self.tick_controls = TickControls(slider=slider, tick_label=tick_label, live_btn=live_btn)

        # Disposition row: Fighter | Scout buttons
        fighter_btn = arcade.gui.UIFlatButton(text="Fighter", width=70, height=20)
        scout_btn = arcade.gui.UIFlatButton(text="Scout", width=70, height=20)

        disposition_row = arcade.gui.UIBoxLayout(vertical=False, space_between=5)
        disposition_row.add(fighter_btn)
        disposition_row.add(scout_btn)
        controls_container.add(disposition_row)

        self.disposition_controls = DispositionControls(fighter_btn=fighter_btn, scout_btn=scout_btn)

        # Anchor the controls below the map
        anchor = arcade.gui.UIAnchorLayout()
        anchor.add(
            controls_container,
            anchor_x="left",
            anchor_y="bottom",
            align_x=self.left,
            align_y=self.bottom - self.map_pixel_size - 60,  # Below map
        )
        self.ui_manager.add(anchor)

        # Register event handlers
        @slider.event("on_change")
        def on_slider_change(event: Any) -> None:
            current_tick = self.client.get_current_tick()
            if current_tick > 0:
                self.view_state.freeze_frame = int(slider.value * current_tick)

        @live_btn.event("on_click")
        def on_live_click(event: Any) -> None:
            self.view_state.freeze_frame = None

        @fighter_btn.event("on_click")
        def on_fighter_click(event: Any) -> None:
            self.client.add_player_action(self.team, CreateUnitPlayerAction(
                mind=PlanningMind(),
                unit_type=UnitType.FIGHTER,
            ))

        @scout_btn.event("on_click")
        def on_scout_click(event: Any) -> None:
            self.client.add_player_action(self.team, CreateUnitPlayerAction(
                mind=PlanningMind(),
                unit_type=UnitType.SCOUT,
            ))

    def _create_plan_controls(self, unit_type: UnitType) -> PlanControls:
        """Create plan control widgets for a selected unit."""
        # Default checkbox states based on unit type
        fight_default, flee_default = (False, True) if unit_type == UnitType.SCOUT else (True, False)

        # Main vertical layout
        main_box = arcade.gui.UIBoxLayout(vertical=True, space_between=5)

        # Selection label
        team_name = self.team.value
        selection_label = arcade.gui.UILabel(
            text=f"Selected: {team_name} unit - click {team_name}'s map to add waypoints",
            font_size=10, text_color=arcade.color.WHITE,
        )
        main_box.add(selection_label)

        # Plan text area
        plan_box_width = self.map_pixel_size
        plan_box_height = 100
        text_area = arcade.gui.UITextArea(
            text="Orders: (none)\nInterrupts:",
            width=plan_box_width, height=plan_box_height,
            font_size=10, text_color=arcade.color.WHITE,
        )
        text_area.with_background(color=arcade.types.Color(40, 40, 40, 255))
        text_area.with_border(color=arcade.color.WHITE, width=2)
        text_area.with_padding(all=5)
        main_box.add(text_area)

        # Button row
        button_row = arcade.gui.UIBoxLayout(vertical=False, space_between=10)
        issue_plan_btn = arcade.gui.UIFlatButton(
            text="Issue Plan", width=80, height=25,
            style=arcade.gui.UIFlatButton.STYLE_BLUE,
        )
        clear_plan_btn = arcade.gui.UIFlatButton(
            text="Clear", width=60, height=25,
            style=arcade.gui.UIFlatButton.STYLE_RED,
        )
        button_row.add(issue_plan_btn)
        button_row.add(clear_plan_btn)
        main_box.add(button_row)

        # Checkbox row
        checkbox_row = arcade.gui.UIBoxLayout(vertical=False, space_between=5)

        def make_checkbox_with_label(label: str, default: bool) -> tuple[arcade.gui.UITextureToggle, arcade.gui.UIBoxLayout]:
            box = arcade.gui.UIBoxLayout(vertical=False, space_between=3)
            checkbox = arcade.gui.UITextureToggle(
                on_texture=TEX_CHECKBOX_CHECKED,
                off_texture=TEX_CHECKBOX_UNCHECKED,
                width=20, height=20, value=default,
            )
            box.add(checkbox)
            box.add(arcade.gui.UILabel(text=label, font_size=10, text_color=arcade.color.WHITE))
            return checkbox, box

        fight_checkbox, fight_box = make_checkbox_with_label("fight", fight_default)
        flee_checkbox, flee_box = make_checkbox_with_label("flee", flee_default)
        forage_checkbox, forage_box = make_checkbox_with_label("forage", True)
        come_back_checkbox, come_back_box = make_checkbox_with_label("come back", True)

        checkbox_row.add(fight_box)
        checkbox_row.add(flee_box)
        checkbox_row.add(forage_box)
        checkbox_row.add(come_back_box)
        main_box.add(checkbox_row)

        # Anchor below disposition controls
        anchor = arcade.gui.UIAnchorLayout()
        anchor.add(
            main_box,
            anchor_x="left",
            anchor_y="bottom",
            align_x=self.left,
            align_y=self.bottom - self.map_pixel_size - 120,  # Below tick/disposition
        )
        self.ui_manager.add(anchor)

        controls = PlanControls(
            container=anchor,
            text_area=text_area,
            issue_plan_btn=issue_plan_btn,
            clear_plan_btn=clear_plan_btn,
            selection_label=selection_label,
            fight_checkbox=fight_checkbox,
            flee_checkbox=flee_checkbox,
            forage_checkbox=forage_checkbox,
            come_back_checkbox=come_back_checkbox,
        )

        # Event handlers
        @issue_plan_btn.event("on_click")
        def on_issue_plan_click(event: Any) -> None:
            if self.view_state.working_plan and self.view_state.working_plan.orders and self.view_state.selected_unit_id:
                self.client.add_player_action(self.team, SetUnitPlanPlayerAction(
                    unit_id=self.view_state.selected_unit_id,
                    plan=self.view_state.working_plan,
                ))
                self._clear_selection()

        @clear_plan_btn.event("on_click")
        def on_clear_plan_click(event: Any) -> None:
            if self.view_state.selected_unit_id:
                knowledge = self.client.get_player_knowledge(self.team, self.client.get_current_tick())
                unit_type = UnitType.FIGHTER
                if self.view_state.selected_unit_id in knowledge.last_in_base:
                    _, selected_unit = knowledge.last_in_base[self.view_state.selected_unit_id]
                    unit_type = selected_unit.unit_type
                self.view_state.working_plan = Plan(interrupts=make_initial_working_plan_interrupts(unit_type))
                if self.plan_controls:
                    if unit_type == UnitType.SCOUT:
                        self.plan_controls.fight_checkbox.value = False
                        self.plan_controls.flee_checkbox.value = True
                    else:
                        self.plan_controls.fight_checkbox.value = True
                        self.plan_controls.flee_checkbox.value = False
                    self.plan_controls.forage_checkbox.value = True
                    self.plan_controls.come_back_checkbox.value = True

        def update_interrupts_from_checkboxes() -> None:
            if self.view_state.working_plan and self.plan_controls:
                self.view_state.working_plan.interrupts = make_interrupts_from_checkboxes(
                    fight=self.plan_controls.fight_checkbox.value,
                    flee=self.plan_controls.flee_checkbox.value,
                    forage=self.plan_controls.forage_checkbox.value,
                    come_back=self.plan_controls.come_back_checkbox.value,
                )

        for checkbox in [fight_checkbox, flee_checkbox, forage_checkbox, come_back_checkbox]:
            @checkbox.event("on_change")
            def on_checkbox_change(event: Any) -> None:
                update_interrupts_from_checkboxes()

        return controls

    def _clear_selection(self) -> None:
        """Clear the current unit selection and plan controls."""
        self.view_state.selected_unit_id = None
        self.view_state.working_plan = None
        if self.plan_controls is not None:
            self.ui_manager.remove(self.plan_controls.container)
            self.plan_controls = None

    def on_draw(self) -> None:
        """Draw the player's map view."""
        freeze_frame = self.view_state.freeze_frame
        knowledge = self.view_state.knowledge
        view_t = knowledge.tick if freeze_frame is None else freeze_frame
        logbook = knowledge.logbook.observation_log

        map_pixel_width = self.grid_width * TILE_SIZE
        map_pixel_height = self.grid_height * TILE_SIZE
        ox, oy = int(self.left), int(self.bottom)

        arcade.draw_lbwh_rectangle_filled(ox, oy, map_pixel_width, map_pixel_height, (0, 0, 0))

        cur_observations = logbook.get(view_t, {})

        # Draw cells with gradient tinting based on observation age
        for pos, (last_observed_tick, _) in knowledge.logbook.last_observations_by_pos.items():
            age = view_t - last_observed_tick
            brightness = int(80 * (2 ** (-(age / CELL_BRIGHTNESS_HALFLIFE))))
            brightness = max(0, min(80, brightness))
            cell_left = ox + pos.x * TILE_SIZE
            cell_bottom = oy + map_pixel_height - (pos.y + 1) * TILE_SIZE
            arcade.draw_lbwh_rectangle_filled(cell_left, cell_bottom, TILE_SIZE, TILE_SIZE, (brightness, brightness, brightness))

        draw_grid(ox, oy, map_pixel_width, map_pixel_height, self.grid_width, self.grid_height)

        if freeze_frame is None:
            for pos, (t, contents_list) in knowledge.logbook.last_observations_by_pos.items():
                for contents in sorted(contents_list, key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent))):
                    if isinstance(contents, BasePresent):
                        draw_base_cell(pos, contents.team, ox, oy, map_pixel_height)
                    elif isinstance(contents, UnitPresent):
                        if pos not in cur_observations and contents.team == self.team:
                            continue
                        draw_unit_at(
                            contents.team, pos, ox, oy, map_pixel_height,
                            outline_only=pos not in cur_observations,
                            unit_type=contents.unit_type,
                        )
                    elif isinstance(contents, FoodPresent):
                        draw_food({pos: contents.count}, ox, oy, map_pixel_height, outline_only=pos not in cur_observations)
        else:
            for pos, contents_list in logbook.get(freeze_frame, {}).items():
                for contents in sorted(contents_list, key=lambda x: (isinstance(x, UnitPresent), isinstance(x, FoodPresent))):
                    if isinstance(contents, BasePresent):
                        draw_base_cell(pos, contents.team, ox, oy, map_pixel_height)
                    elif isinstance(contents, UnitPresent):
                        draw_unit_at(contents.team, pos, ox, oy, map_pixel_height, unit_type=contents.unit_type)
                    elif isinstance(contents, FoodPresent):
                        draw_food({pos: contents.count}, ox, oy, map_pixel_height)

        # Draw predicted positions
        for unit_id, trajectory in knowledge.expected_trajectories.items():
            predicted_pos = trajectory.get(view_t)
            if predicted_pos is None:
                continue
            unit_type = knowledge.last_in_base[unit_id][1].unit_type
            draw_unit_at(self.team, predicted_pos, ox, oy, map_pixel_height, outline_only=True, unit_type=unit_type)

    def on_update(self, delta_time: float) -> None:
        """Update UI state."""
        # Sync knowledge from lifecycle
        self.view_state.knowledge = self.lifecycle.knowledge[self.team]

        # Update tick controls
        current_tick = self.client.get_current_tick()
        if current_tick > 0 and self.tick_controls:
            if self.view_state.freeze_frame is None:
                self.tick_controls.slider.value = 1.0
                self.tick_controls.tick_label.text = f"t={current_tick}"
            else:
                self.tick_controls.slider.value = self.view_state.freeze_frame / current_tick
                self.tick_controls.tick_label.text = f"t={self.view_state.freeze_frame}"

        # Update disposition button text
        if self.disposition_controls:
            food_count = self.client.get_food_count_in_base(self.team)
            has_food = food_count > 0
            self.disposition_controls.fighter_btn.text = f"Fighter ({food_count})"
            self.disposition_controls.scout_btn.text = f"Scout ({food_count})"
            self.disposition_controls.fighter_btn.disabled = not has_food
            self.disposition_controls.scout_btn.disabled = not has_food

        # Handle plan controls
        if self.view_state.selected_unit_id is not None:
            knowledge = self.client.get_player_knowledge(self.team, self.client.get_current_tick())
            selected_unit = knowledge.own_units_in_base.get(self.view_state.selected_unit_id)

            if selected_unit is None:
                self._clear_selection()
            elif self.plan_controls is None and self.view_state.working_plan is not None:
                self.plan_controls = self._create_plan_controls(selected_unit.unit_type)

            if self.view_state.working_plan is not None and self.plan_controls is not None and selected_unit is not None:
                plan_lines = format_plan(self.view_state.working_plan, selected_unit)
                self.plan_controls.text_area.text = "\n".join(plan_lines)
        else:
            if self.plan_controls is not None:
                self.ui_manager.remove(self.plan_controls.container)
                self.plan_controls = None

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> bool:
        """Handle mouse clicks on the map."""
        if button != arcade.MOUSE_BUTTON_LEFT:
            return False

        # Only allow interaction when live
        if self.view_state.freeze_frame is not None:
            return False

        grid_pos = screen_to_grid(x, y, self.grid_width, self.grid_height, self.map_height)
        if grid_pos is None:
            return False

        current_elapsed = self.start_time_getter()
        is_double_click = (
            self.view_state.last_click_pos == grid_pos
            and current_elapsed - self.view_state.last_click_time < 0.3
        )
        self.view_state.last_click_time = current_elapsed
        self.view_state.last_click_pos = grid_pos

        if is_double_click and self.view_state.selected_unit_id is not None:
            if self.view_state.working_plan is not None and self.view_state.working_plan.orders:
                self.client.add_player_action(self.team, SetUnitPlanPlayerAction(
                    unit_id=self.view_state.selected_unit_id,
                    plan=self.view_state.working_plan,
                ))
                self._clear_selection()
        elif self.view_state.selected_unit_id is not None:
            if self.view_state.working_plan is None:
                knowledge = self.client.get_player_knowledge(self.team, self.client.get_current_tick())
                unit_type = UnitType.FIGHTER
                if self.view_state.selected_unit_id in knowledge.last_in_base:
                    _, selected_unit = knowledge.last_in_base[self.view_state.selected_unit_id]
                    unit_type = selected_unit.unit_type
                self.view_state.working_plan = Plan(interrupts=make_initial_working_plan_interrupts(unit_type))
            self.view_state.working_plan.orders.append(Move(target=grid_pos))
        else:
            knowledge = self.client.get_player_knowledge(self.team, self.client.get_current_tick())
            for unit_id, unit in knowledge.own_units_in_base.items():
                if unit.pos == grid_pos:
                    self.view_state.selected_unit_id = unit_id
                    self.view_state.working_plan = Plan(interrupts=make_initial_working_plan_interrupts(unit.unit_type))
                    break

        return True


class AntGameView(arcade.View):
    """Main view containing all game sections."""

    def __init__(self, lifecycle: GameLifecycle):
        super().__init__()
        self.lifecycle = lifecycle
        self.start_at = time.time()

        self.ui_manager = arcade.gui.UIManager()
        self.section_manager = SectionManager(self)

        # Create view states for each team
        self.view_states = {
            team: PlayerViewState(knowledge=lifecycle.knowledge[team])
            for team in Team
        }

        # Sections will be created in setup
        self.red_section: PlayerMapSection | None = None
        self.god_section: GodMapSection | None = None
        self.blue_section: PlayerMapSection | None = None

    def get_elapsed_secs(self) -> float:
        return time.time() - self.start_at

    def setup(self) -> None:
        """Set up the view with sections."""
        grid_width = self.lifecycle.grid_width
        grid_height = self.lifecycle.grid_height
        map_pixel_size = grid_width * TILE_SIZE
        control_area_height = 200

        # Calculate section positions
        red_left = PADDING
        god_left = PADDING * 2 + map_pixel_size
        blue_left = PADDING * 3 + map_pixel_size * 2
        section_bottom = control_area_height + PADDING

        # Create sections
        self.red_section = PlayerMapSection(
            left=red_left,
            bottom=section_bottom,
            width=map_pixel_size,
            height=map_pixel_size,
            team=Team.RED,
            lifecycle=self.lifecycle,
            view_state=self.view_states[Team.RED],
            ui_manager=self.ui_manager,
            start_time_getter=self.get_elapsed_secs,
        )

        self.god_section = GodMapSection(
            left=god_left,
            bottom=section_bottom,
            width=map_pixel_size,
            height=map_pixel_size,
            lifecycle=self.lifecycle,
        )

        self.blue_section = PlayerMapSection(
            left=blue_left,
            bottom=section_bottom,
            width=map_pixel_size,
            height=map_pixel_size,
            team=Team.BLUE,
            lifecycle=self.lifecycle,
            view_state=self.view_states[Team.BLUE],
            ui_manager=self.ui_manager,
            start_time_getter=self.get_elapsed_secs,
        )

        # Add sections to the view
        self.section_manager.add_section(self.red_section)
        self.section_manager.add_section(self.god_section)
        self.section_manager.add_section(self.blue_section)

    def on_show_view(self) -> None:
        self.ui_manager.enable()

    def on_hide_view(self) -> None:
        self.ui_manager.disable()

    def on_draw(self) -> None:
        self.clear()
        # Sections draw themselves via section_manager
        self.ui_manager.draw()

    def on_update(self, delta_time: float) -> None:
        self.ui_manager.on_update(delta_time)  # type: ignore[no-untyped-call]

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> bool:
        # Let UI manager handle it first
        if self.ui_manager.on_mouse_press(x, y, button, modifiers):
            return True
        return False


def create_window_from_lifecycle(lifecycle: GameLifecycle) -> tuple[AntGameView, arcade.Window]:
    """Create UI window and view from a GameLifecycle."""
    grid_width = lifecycle.grid_width
    grid_height = lifecycle.grid_height
    map_pixel_size = grid_width * TILE_SIZE
    control_area_height = 200

    window_width = map_pixel_size * 3 + PADDING * 4
    window_height = map_pixel_size + PADDING * 2 + control_area_height

    window = arcade.Window(window_width, window_height, "Ant RTS")
    window.set_update_rate(1/60)

    view = AntGameView(lifecycle)
    view.setup()
    window.show_view(view)

    return view, window


def main() -> None:
    """Main entry point: parse args, create lifecycle, and run UI."""
    args = parse_args()
    lifecycle = create_lifecycle_from_args(args)
    view, window = create_window_from_lifecycle(lifecycle)
    arcade.run()


if __name__ == "__main__":
    main()
