
from core import Pos
from mechanics import Team, Unit, generate_unit_id
from planning import Plan


def make_unit(
    team: Team,
    pos: Pos,  # pyright: ignore[reportUndefinedVariable]
    original_pos: Pos | None = None,
) -> Unit:
    """Create a unit with an auto-generated ID. Useful for tests."""
    return Unit(
        id=generate_unit_id(),
        team=team,
        pos=pos,
        original_pos=original_pos if original_pos is not None else pos,
    )
