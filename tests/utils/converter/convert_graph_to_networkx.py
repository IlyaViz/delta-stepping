import networkx as nx
from src.weighted_graph.weighted_graph import WeightedGraph


def convert_graph_to_networkx(graph: WeightedGraph) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for vertex in graph.vertices:
        G.add_node(vertex.id)

    for vertex in graph.vertices:
        for edge in graph.get_vertex_edges(vertex):
            neighbour_vertex = graph.get_vertex_neighbour_by_edge(vertex, edge)
            G.add_edge(vertex.id, neighbour_vertex.id, weight=edge.weight)

    return G
