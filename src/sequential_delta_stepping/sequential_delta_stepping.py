import copy


def sequential_delta_stepping(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex: int,
    delta: float,
) -> list[float]:
    def relax_neighbour(vertex: int, edge_index: int) -> bool:
        neighbour_vertex = neighbours[vertex][edge_index]
        new_distance = distances[vertex] + weights[vertex][edge_index]

        if new_distance < distances[neighbour_vertex]:
            distances[neighbour_vertex] = new_distance

            return True

        return False

    def add_to_bucket(vertex: int, buckets: list[set]) -> None:
        bucket_index = int(distances[vertex] // delta)
        buckets_length = len(buckets)

        for _ in range(bucket_index - buckets_length + 1):
            buckets.append(set())

        buckets[bucket_index].add(vertex)

    vertices_count = len(neighbours)
    distances = [float("inf")] * vertices_count

    distances[source_vertex] = 0

    buckets = []
    current_heavy_edges = set()
    current_bucket_index = 0

    add_to_bucket(source_vertex, buckets)

    while current_bucket_index < len(buckets):
        current_bucket = buckets[current_bucket_index]

        for vertex in tuple(current_bucket):
            current_bucket.remove(vertex)

            if int(distances[vertex] // delta) != current_bucket_index:
                continue

            for edge_index in range(len(neighbours[vertex])):
                edge = weights[vertex][edge_index]
                is_light_edge = edge <= delta

                if not is_light_edge:
                    current_heavy_edges.add((vertex, edge_index))
                    continue

                if relax_neighbour(vertex, edge_index):
                    neighbour_vertex = neighbours[vertex][edge_index]
                    add_to_bucket(neighbour_vertex, buckets)

        if not current_bucket:
            for heavy_edge in current_heavy_edges:
                vertex, edge_index = heavy_edge

                if relax_neighbour(vertex, edge_index):
                    neighbour_vertex = neighbours[vertex][edge_index]
                    add_to_bucket(neighbour_vertex, buckets)

            current_heavy_edges.clear()

            current_bucket_index += 1

    return distances
