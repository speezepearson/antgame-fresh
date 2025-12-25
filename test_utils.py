
from core import Pos
from mechanics import Team, Unit, generate_unit_id
from planning import Plan, PlanningMind


def make_unit(
    team: Team,
    pos: Pos,
    plan: Plan | None = None,
) -> Unit:
    """Create a unit with an auto-generated ID and PlanningMind. Useful for tests."""
    mind = PlanningMind(plan=plan if plan is not None else Plan(), original_pos=pos)
    return Unit(
        id=generate_unit_id(),
        team=team,
        mind=mind,
        pos=pos,
    )
