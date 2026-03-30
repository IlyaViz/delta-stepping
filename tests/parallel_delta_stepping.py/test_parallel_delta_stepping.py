import math
from networkx import single_source_dijkstra_path_length
from src.parallel_delta_stepping.parallel_delta_stepping import parallel_delta_stepping
from src.const.type_error import ABS_TOL, REL_TOL
from tests.const import (
    TEST_GRAPH_VERTICES_COUNT,
    TEST_DELTA,
    TEST_CPU_COUNT,
)


def test_parallel_delta_stepping(test_graph):
    neighbours, weights, graphx = test_graph

    source_vertex = 0

    distances = parallel_delta_stepping(
        neighbours, weights, source_vertex, TEST_DELTA, TEST_CPU_COUNT
    )
    expected_distances = single_source_dijkstra_path_length(
        graphx, source_vertex, weight="weight"
    )

    for vertex in range(TEST_GRAPH_VERTICES_COUNT):
        assert math.isclose(
            distances[vertex],
            expected_distances.get(vertex, float("inf")),
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        ), f"Distance to vertex {vertex} is incorrect. Expected: {expected_distances.get(vertex, float('inf'))}, Got: {distances[vertex]}"
