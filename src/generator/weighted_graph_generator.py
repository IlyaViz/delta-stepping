import random


def generate_weighted_graph_with_default_types(
    num_vertices: int, num_edges: int, min_weight: int, max_weight: int
) -> tuple[list[list[int]], list[list[float]]]:
    neighbours = [[] for _ in range(num_vertices)]
    weights = [[] for _ in range(num_vertices)]

    for _ in range(num_edges):
        source = random.randint(0, num_vertices - 1)
        target = random.randint(0, num_vertices - 1)

        while target == source:
            target = random.randint(0, num_vertices - 1)

        weight = random.uniform(min_weight, max_weight)
        neighbours[source].append(target)
        weights[source].append(weight)

    return neighbours, weights
