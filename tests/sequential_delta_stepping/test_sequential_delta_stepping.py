import math
from networkx import single_source_dijkstra_path_length
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.const.type_error import ABS_TOL, REL_TOL
from tests.const import (
    TEST_GRAPH_VERTICES_COUNT,
    TEST_DELTA,
)


def test_sequential_delta_stepping(test_graph):
    neighbours, weights, graphx = test_graph
    
    source_vertex = 0

    distances = sequential_delta_stepping(
        neighbours, weights, source_vertex, TEST_DELTA
    )
    expected_distances = single_source_dijkstra_path_length(
        graphx, source_vertex, weight="weight"
    )

    for vertex in range(TEST_GRAPH_VERTICES_COUNT):
        assert math.isclose(
            distances[vertex], expected_distances[vertex], abs_tol=ABS_TOL, rel_tol=REL_TOL
        )
