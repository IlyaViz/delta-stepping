import pytest
import math
from networkx import single_source_dijkstra_path_length
from src.parallel_delta_stepping.parallel_delta_stepping import parallel_delta_stepping
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.const.type_error import ABS_TOL, REL_TOL


@pytest.mark.parametrize("algo", [sequential_delta_stepping, parallel_delta_stepping])
def test_delta_stepping(test_graphs, algo):
    for neighbours, weights, graphx in zip(*test_graphs):
        source_vertex = 0

        distances = algo(neighbours, weights, source_vertex, -1)
        expected_distances = single_source_dijkstra_path_length(
            graphx, source_vertex, weight="weight"
        )

        for vertex in range(len(distances)):
            assert math.isclose(
                distances[vertex],
                expected_distances.get(vertex, float("inf")),
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            ), f"Distance to vertex {vertex} is incorrect. Expected: {expected_distances.get(vertex, float('inf'))}, Got: {distances[vertex]}"
