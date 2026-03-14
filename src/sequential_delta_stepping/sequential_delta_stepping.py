import copy
from src.weighted_graph.weighted_graph import WeightedGraph
from src.weighted_graph.weighted_vertex import WeightedVertex
from src.weighted_graph.weighted_edge import WeightedEdge


def sequential_delta_stepping(
    graph: WeightedGraph,
    source_vertex: WeightedVertex,
    delta: float,
) -> None:
    def relax_neighbour(vertex: WeightedVertex, edge: WeightedEdge) -> bool:
        neighbour_vertex = graph.get_vertex_neighbour_by_edge(vertex, edge)
        new_distance = vertex.distance + edge.weight

        if new_distance < neighbour_vertex.distance:
            neighbour_vertex.distance = new_distance
            return True

        return False

    def add_to_bucket(vertex: WeightedVertex, buckets: list[set]) -> None:
        bucket_index = int(vertex.distance // delta)
        buckets_length = len(buckets)

        for _ in range(bucket_index - buckets_length + 1):
            buckets.append(set())

        buckets[bucket_index].add(vertex)

    for vertex in graph.vertices:
        vertex.distance = float("inf")

    source_vertex.distance = 0

    buckets = []
    current_heavy_edges = set()
    current_bucket_index = 0

    add_to_bucket(source_vertex, buckets)

    while current_bucket_index < len(buckets):
        current_bucket = buckets[current_bucket_index]

        for vertex in tuple(current_bucket):
            current_bucket.remove(vertex)

            if int(vertex.distance // delta) != current_bucket_index:
                continue

            vertex_edges = graph.get_vertex_edges(vertex)

            for edge in vertex_edges:
                is_light_edge = edge.weight <= delta

                if not is_light_edge:
                    current_heavy_edges.add((vertex, edge))
                    continue

                neighbour_vertex = graph.get_vertex_neighbour_by_edge(vertex, edge)

                if relax_neighbour(vertex, edge):
                    add_to_bucket(neighbour_vertex, buckets)

        if not current_bucket:
            for heavy_edge in current_heavy_edges:
                vertex, edge = heavy_edge

                neighbour_vertex = graph.get_vertex_neighbour_by_edge(vertex, edge)

                if relax_neighbour(vertex, edge):
                    add_to_bucket(neighbour_vertex, buckets)

            current_heavy_edges.clear()

            current_bucket_index += 1
