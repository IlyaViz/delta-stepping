import pytest
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from tests.const import (
    TEST_GRAPH_VERTICES_OPTIONS,
    TEST_GRAPH_EDGES_OPTIONS,
    TEST_MIN_WEIGHT,
    TEST_MAX_WEIGHT,
)
from tests.test_utils.converter.convert_graph_to_networkx import (
    convert_graph_to_networkx,
)


@pytest.fixture(scope="function")
def test_graphs():
    graphs_neighbours = []
    graphs_weights = []
    graphs_graphxs = []

    for vertices in TEST_GRAPH_VERTICES_OPTIONS:
        for edges in TEST_GRAPH_EDGES_OPTIONS:
            neighbours, weights = generate_weighted_graph_with_default_types(
                vertices, edges, TEST_MIN_WEIGHT, TEST_MAX_WEIGHT
            )
            graphx = convert_graph_to_networkx(neighbours, weights)

            graphs_neighbours.append(neighbours)
            graphs_weights.append(weights)
            graphs_graphxs.append(graphx)

    return graphs_neighbours, graphs_weights, graphs_graphxs
