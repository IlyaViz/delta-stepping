from src.weighted_graph.weighted_vertex import WeightedVertex
from src.weighted_graph.weighted_edge import WeightedEdge


class WeightedGraph:
    cached_vertex_edges = {}

    def __init__(self, vertices: list[WeightedVertex], edges: list[WeightedEdge]):
        self.vertices = vertices
        self.edges = edges

    def get_vertex_edges(self, vertex: WeightedVertex) -> list[WeightedEdge]:
        if vertex in self.cached_vertex_edges:
            return self.cached_vertex_edges[vertex]

        vertex_edges = [edge for edge in self.edges if edge.source == vertex]

        self.cached_vertex_edges[vertex] = vertex_edges

        return vertex_edges

    def get_vertex_neighbour_by_edge(
        self, vertex: WeightedVertex, edge: WeightedEdge
    ) -> WeightedVertex:
        if edge.source == vertex:
            return edge.target
        else:
            return edge.source
