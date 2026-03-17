import networkx as nx


def convert_graph_to_networkx(
    neighbours: list[list[int]], weights: list[list[float]]
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for i in range(len(neighbours)):
        G.add_node(i)

    for i in range(len(neighbours)):
        for j in range(len(neighbours[i])):
            G.add_edge(i, neighbours[i][j], weight=weights[i][j])

    return G
