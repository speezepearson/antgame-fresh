"""Core data structures and utilities."""

from __future__ import annotations
from dataclasses import dataclass


Timestamp = int


@dataclass(frozen=True)
class Pos:
    x: int
    y: int

    def manhattan_distance(self, other: Pos) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass(frozen=True)
class Region:
    """A set of positions."""

    cells: frozenset[Pos]

    def contains(self, pos: Pos) -> bool:
        """Check if position is inside the region."""
        return pos in self.cells

    def get_edge_cells(self) -> list[Pos]:
        """Return perimeter cells (cells with at least one neighbor outside the region)."""
        edge_cells = []
        for cell in self.cells:
            # Check if any orthogonal neighbor is outside the region
            neighbors = [
                Pos(cell.x + 1, cell.y),
                Pos(cell.x - 1, cell.y),
                Pos(cell.x, cell.y + 1),
                Pos(cell.x, cell.y - 1),
            ]
            if any(neighbor not in self.cells for neighbor in neighbors):
                edge_cells.append(cell)
        return edge_cells
