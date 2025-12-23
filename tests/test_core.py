"""Tests for core data structures."""

from core import Pos, Region


class TestManhattanDistance:
    def test_returns_zero_for_same_position(self) -> None:
        p = Pos(5, 5)
        assert p.manhattan_distance(p) == 0

    def test_calculates_horizontal_distance(self) -> None:
        p1 = Pos(0, 5)
        p2 = Pos(7, 5)
        assert p1.manhattan_distance(p2) == 7

    def test_calculates_vertical_distance(self) -> None:
        p1 = Pos(5, 0)
        p2 = Pos(5, 4)
        assert p1.manhattan_distance(p2) == 4

    def test_calculates_diagonal_distance(self) -> None:
        p1 = Pos(0, 0)
        p2 = Pos(3, 4)
        assert p1.manhattan_distance(p2) == 7


class TestRegionContains:
    def test_returns_true_for_position_inside_region(self) -> None:
        cells = frozenset([Pos(0, 0), Pos(1, 0), Pos(0, 1)])
        region = Region(cells)
        assert region.contains(Pos(0, 0))

    def test_returns_false_for_position_outside_region(self) -> None:
        cells = frozenset([Pos(0, 0), Pos(1, 0)])
        region = Region(cells)
        assert not region.contains(Pos(5, 5))


class TestRegionEdgeCells:
    def test_single_cell_is_an_edge(self) -> None:
        region = Region(frozenset([Pos(0, 0)]))
        edges = region.get_edge_cells()
        assert len(edges) == 1
        assert Pos(0, 0) in edges

    def test_two_by_two_square_has_all_cells_as_edges(self) -> None:
        cells = frozenset([Pos(x, y) for x in range(2) for y in range(2)])
        region = Region(cells)
        edges = region.get_edge_cells()
        assert len(edges) == 4

    def test_three_by_three_square_has_only_outer_cells_as_edges(self) -> None:
        cells = frozenset([Pos(x, y) for x in range(3) for y in range(3)])
        region = Region(cells)
        edges = region.get_edge_cells()
        # Center cell (1,1) should not be an edge
        assert len(edges) == 8
        assert Pos(1, 1) not in edges

    def test_l_shape_identifies_correct_edges(self) -> None:
        # L shape: (0,0), (0,1), (0,2), (1,2)
        cells = frozenset([Pos(0, 0), Pos(0, 1), Pos(0, 2), Pos(1, 2)])
        region = Region(cells)
        edges = region.get_edge_cells()
        # All should be edges since none has all 4 neighbors inside
        assert len(edges) == 4
