from src.weighted_graph.weighted_graph import WeightedGraph
from src.weighted_graph.weighted_vertex import WeightedVertex
from src.weighted_graph.weighted_edge import WeightedEdge

def generate_weighted_graph(num_vertices: int, num_edges: int, only_positive_weights: bool = True) -> WeightedGraph:
    vertices = [WeightedVertex(i) for i in range(num_vertices)]
    edges = []

    for i in range(num_edges):
        source_index = i % num_vertices
        target_index = (i + 1) % num_vertices

        weight = (i + 1) if only_positive_weights else (i - num_edges // 2)

        edge = WeightedEdge(vertices[source_index], vertices[target_index], weight)
        edges.append(edge)

    return WeightedGraph(vertices, edges)