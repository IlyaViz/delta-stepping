import math
from networkx import single_source_dijkstra_path_length
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from tests.utils.converter.convert_graph_to_networkx import convert_graph_to_networkx
from tests.const import (
    TEST_GRAPH_VERTICES_COUNT,
    TEST_GRAPH_EDGES_COUNT,
    TEST_DELTA,
    MIN_WEIGHT,
    MAX_WEIGHT,
)


def test_sequential_delta_stepping():
    neighbours, weights = generate_weighted_graph_with_default_types(
        TEST_GRAPH_VERTICES_COUNT, TEST_GRAPH_EDGES_COUNT, MIN_WEIGHT, MAX_WEIGHT
    )
    graphx = convert_graph_to_networkx(neighbours, weights)
    source_vertex = 0

    distances = sequential_delta_stepping(
        neighbours, weights, source_vertex, TEST_DELTA
    )
    expected_distances = single_source_dijkstra_path_length(
        graphx, source_vertex, weight="weight"
    )

    for vertex in range(TEST_GRAPH_VERTICES_COUNT):
        assert math.isclose(
            distances[vertex], expected_distances[vertex], abs_tol=1e-9, rel_tol=1e-9
        )
