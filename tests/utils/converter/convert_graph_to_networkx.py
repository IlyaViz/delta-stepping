import networkx as nx
from src.weighted_graph.weighted_graph import WeightedGraph


def convert_graph_to_networkx(graph: WeightedGraph) -> nx.DiGraph:
    G = nx.DiGraph()

    for vertex in graph.vertices:
        G.add_node(vertex.id)

    for edge in graph.edges:
        u = edge.source.id
        v = edge.target.id

        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], edge.weight)
        else:
            G.add_edge(u, v, weight=edge.weight)

    return G
