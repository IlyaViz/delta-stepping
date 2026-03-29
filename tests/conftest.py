import pytest
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from tests.const import (
    TEST_GRAPH_VERTICES_COUNT,
    TEST_GRAPH_EDGES_COUNT,
    MIN_WEIGHT,
    MAX_WEIGHT,
)
from tests.test_utils.converter.convert_graph_to_networkx import (
    convert_graph_to_networkx,
)


@pytest.fixture(scope="function")
def test_graph():
    neighbours, weights = generate_weighted_graph_with_default_types(
        TEST_GRAPH_VERTICES_COUNT, TEST_GRAPH_EDGES_COUNT, MIN_WEIGHT, MAX_WEIGHT
    )
    graphx = convert_graph_to_networkx(neighbours, weights)

    return neighbours, weights, graphx
