from networkx import single_source_dijkstra_path_length
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.generator.weighted_graph_generator import generate_weighted_graph
from tests.utils.converter.convert_graph_to_networkx import convert_graph_to_networkx
from tests.const import TEST_GRAPH_VERTICES_COUNT, TEST_GRAPH_EDGES_COUNT, TEST_DELTA


def test_sequential_delta_stepping():
    graph = generate_weighted_graph(TEST_GRAPH_VERTICES_COUNT, TEST_GRAPH_EDGES_COUNT)
    graphx = convert_graph_to_networkx(graph)
    source_vertex = graph.vertices[0]

    sequential_delta_stepping(graph, source_vertex, TEST_DELTA)
    expected_distances = single_source_dijkstra_path_length(
        graphx, source_vertex.id, weight="weight"
    )

    for vertex in graph.vertices:
        assert vertex.distance == expected_distances.get(vertex.id, float("inf"))
