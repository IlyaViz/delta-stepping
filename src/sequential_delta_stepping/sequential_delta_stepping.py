from numpy import ndarray
from src.shared_delta_stepping.shared_delta_stepping_functions import (
    add_to_bucket,
    validate_and_prepare_variables,
    prepare_ndarrays,
)
from src.const.dtype import INT_TYPE, FLOAT_TYPE


def sequential_delta_stepping(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex_index: int,
    delta: float = -1,
) -> list[float]:
    (
        vertices_length,
        max_degree,
        max_buckets,
        delta,
    ) = validate_and_prepare_variables(neighbours, weights, source_vertex_index, delta)

    neighbours, distances, previous, weights, buckets, in_bucket, bucket_sizes = (
        prepare_ndarrays(
            vertices_length,
            max_degree,
            max_buckets,
            set_defaults=True,
            neighbours=neighbours,
            weights=weights,
            source_vertex_index=source_vertex_index,
        )
    )

    add_to_bucket(
        delta,
        source_vertex_index,
        distances,
        buckets,
        bucket_sizes,
        max_buckets,
    )

    next_non_empty_bucket_absolute_index = 0
    next_non_empty_bucket_actual_index = (
        next_non_empty_bucket_absolute_index % max_buckets
    )

    while next_non_empty_bucket_absolute_index != -1:
        process_bucket(
            next_non_empty_bucket_actual_index,
            max_degree,
            max_buckets,
            delta,
            neighbours,
            distances,
            previous,
            weights,
            buckets,
            in_bucket,
            bucket_sizes,
        )

        current_absolute_index = next_non_empty_bucket_absolute_index
        next_non_empty_bucket_absolute_index = -1

        for i in range(max_buckets):
            absolute_index = current_absolute_index + i
            actual_index = absolute_index % max_buckets

            if bucket_sizes[actual_index] > 0:
                next_non_empty_bucket_absolute_index = absolute_index
                next_non_empty_bucket_actual_index = actual_index
                break

    return distances.tolist(), previous.tolist()


def process_bucket(
    actual_bucket_index: int,
    max_degree: int,
    max_buckets: int,
    delta: float,
    neighbours: ndarray[INT_TYPE],
    distances: ndarray[FLOAT_TYPE],
    previous: ndarray[INT_TYPE],
    weights: ndarray[FLOAT_TYPE],
    buckets: ndarray[INT_TYPE],
    in_bucket: ndarray[bool],
    bucket_sizes: ndarray[INT_TYPE],
) -> None:
    heavy_edges = set()

    while bucket_sizes[actual_bucket_index] > 0:
        last_vertex_index = bucket_sizes[actual_bucket_index] - 1
        vertex_index = buckets[actual_bucket_index, last_vertex_index]

        in_bucket[actual_bucket_index, vertex_index] = False

        bucket_sizes[actual_bucket_index] -= 1

        if int(distances[vertex_index] // delta) % max_buckets != actual_bucket_index:
            continue

        vertex_neighbour_indexes = neighbours[vertex_index]

        for i in range(max_degree):
            neighbour_index = vertex_neighbour_indexes[i]

            if neighbour_index == -1:
                break

            edge_weight = weights[vertex_index, i]
            is_light_edge = edge_weight <= delta

            if not is_light_edge:
                heavy_edges.add((vertex_index, neighbour_index, edge_weight))
                continue

            if relax_neighbour(
                vertex_index,
                neighbour_index,
                edge_weight,
                distances,
                previous,
            ):
                bucket_index = int(distances[neighbour_index] // delta) % max_buckets

                if not in_bucket[bucket_index, neighbour_index]:
                    in_bucket[bucket_index, neighbour_index] = True

                    add_to_bucket(
                        delta,
                        neighbour_index,
                        distances,
                        buckets,
                        bucket_sizes,
                        max_buckets,
                    )

    for vertex_index, neighbour_index, edge_weight in heavy_edges:
        if relax_neighbour(
            vertex_index,
            neighbour_index,
            edge_weight,
            distances,
            previous,
        ):
            bucket_index = int(distances[neighbour_index] // delta) % max_buckets

            if not in_bucket[bucket_index, neighbour_index]:
                in_bucket[bucket_index, neighbour_index] = True

                add_to_bucket(
                    delta,
                    neighbour_index,
                    distances,
                    buckets,
                    bucket_sizes,
                    max_buckets,
                )


def relax_neighbour(
    vertex_index: int,
    neighbour_index: int,
    edge_weight: float,
    distances: ndarray[FLOAT_TYPE],
    previous: ndarray[INT_TYPE], 
) -> bool:
    new_distance = distances[vertex_index] + edge_weight

    if new_distance < distances[neighbour_index]:
        distances[neighbour_index] = new_distance
        previous[neighbour_index] = vertex_index

        return True

    return False
