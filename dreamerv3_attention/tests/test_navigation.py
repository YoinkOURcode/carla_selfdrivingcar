import pytest

from dreamerv3_attention.navigation import Route


def test_route_filters_unrelated_crossings_and_handles_crossing_behind():
    route = Route([(0, 0), (10, 0), (20, 0), (30, 0)])
    crossings = route.crossings([[(8, -2), (12, -2), (12, 2), (8, 2)],
                                 [(15, 10), (18, 10), (18, 13), (15, 13)]])
    assert crossings == [("0", 8., 12.)]
    progress, lateral, heading = route.project((16, 1))
    assert progress == 16 and lateral == 1 and heading == 0
    assert crossings[0][2] < progress


def test_tangent_vertex_does_not_count_as_crossing():
    route = Route([(0, 0), (20, 0)])
    assert route.crossings([[(10, 0), (12, 2), (8, 2)]]) == []


def test_route_rejects_degenerate_and_preserves_curve_projection():
    with pytest.raises(ValueError):
        Route([(0, 0), (0, 0)])
    route = Route([(0, 0), (10, 0), (10, 10)])
    progress, lateral, heading = route.project((11, 5))
    assert progress == 15 and lateral == -1
    assert heading == pytest.approx(1.57079632679)


def test_route_lane_metadata_follows_duplicate_filter_and_junction_edges():
    route = Route([(0, 0), (0, 0), (10, 0), (10, 10), (20, 10)],
                  lane_widths=[4., 99., 4., 4., 2.],
                  junctions=[False, False, True, True, False])
    assert route.lane_geometry(5., 3.) == (4., True)
    assert route.lane_geometry(15., 3.) == (4., True)
    width, junction = route.lane_geometry(25., 3.)
    assert width == 3. and junction
    assert route.lane_geometry(30., 3.) == (2., True)


def test_route_projection_preserves_nearby_branch_at_intersection():
    route = Route([(-10, 0), (0, 0), (10, 0), (10, 10), (0, 10), (0, -10)])
    route.project((-5., 0.))
    progress, lateral, heading = route.project((0., 0.))
    assert progress == 10. and lateral == 0. and heading == 0.
    prior_index = route.index
    route.project((0., 9.), local=False)
    assert route.index == prior_index


@pytest.mark.parametrize("name,values", [("lane_widths", [3.]), ("junctions", [True])])
def test_route_rejects_misaligned_lane_metadata(name, values):
    with pytest.raises(ValueError):
        Route([(0, 0), (10, 0)], **{name: values})
