from numpy import ndarray
from src.const.dtype import FLOAT_DTYPE, INT_DTYPE, INT_TYPE, FLOAT_TYPE
from src.utils.delta_stepping_params_validator import validate_delta_stepping_params


def validate_and_prepare_variables(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex_index: int,
    delta: float,
) -> tuple[
    int, int, int, float, tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]
]:
    validate_delta_stepping_params(neighbours, weights, source_vertex_index, delta)

    vertices_length = len(neighbours)
    max_degree = max(len(neighbour) for neighbour in neighbours)
    flattened_weights = [weight for sublist in weights for weight in sublist]

    if delta == -1:
        avg_weight = sum(flattened_weights) / len(flattened_weights)
        delta = avg_weight / 25

    max_weight = max(flattened_weights)
    max_buckets = int(max_weight // delta) + 2

    return vertices_length, max_degree, max_buckets, delta


def prepare_ndarrays(
    vertices_length: int,
    max_degree: int,
    max_buckets: int,
    buffers: list[memoryview[int]] | None = None,
    set_defaults: bool = False,
    source_vertex_index: int = 0,
    neighbours: list[list[int]] = None,
    weights: list[list[float]] = None,
) -> tuple[
    ndarray[INT_TYPE],
    ndarray[FLOAT_TYPE],
    ndarray[FLOAT_TYPE],
    ndarray[INT_TYPE],
    ndarray[bool],
    ndarray[INT_TYPE],
]:
    shared_neighbours = ndarray(
        (vertices_length, max_degree),
        dtype=INT_DTYPE,
        buffer=buffers[0] if buffers else None,
    )
    shared_distances = ndarray(
        (vertices_length,),
        dtype=FLOAT_DTYPE,
        buffer=buffers[1] if buffers else None,
    )
    shared_weights = ndarray(
        (vertices_length, max_degree),
        dtype=FLOAT_DTYPE,
        buffer=buffers[2] if buffers else None,
    )
    shared_buckets = ndarray(
        (max_buckets, vertices_length),
        dtype=INT_DTYPE,
        buffer=buffers[3] if buffers else None,
    )
    shared_in_bucket = ndarray(
        (max_buckets, vertices_length),
        dtype=bool,
        buffer=buffers[4] if buffers else None,
    )
    shared_bucket_sizes = ndarray(
        (max_buckets,),
        dtype=INT_DTYPE,
        buffer=buffers[5] if buffers else None,
    )

    if set_defaults:
        shared_neighbours.fill(-1)
        shared_distances.fill(float("inf"))
        shared_in_bucket.fill(False)
        shared_bucket_sizes.fill(0)

        shared_distances[source_vertex_index] = 0

        padded_neighbours = [row + [-1] * (max_degree - len(row)) for row in neighbours]
        padded_weights = [row + [-1.0] * (max_degree - len(row)) for row in weights]

        shared_neighbours[:] = padded_neighbours
        shared_weights[:] = padded_weights

    return (
        shared_neighbours,
        shared_distances,
        shared_weights,
        shared_buckets,
        shared_in_bucket,
        shared_bucket_sizes,
    )


def add_to_bucket(
    delta: float,
    vertex_index: int,
    distances: ndarray[FLOAT_TYPE],
    buckets: ndarray[INT_TYPE],
    bucket_sizes: ndarray[INT_TYPE],
    max_buckets: int,
) -> None:
    bucket_index = int(distances[vertex_index] // delta) % max_buckets

    current_size = bucket_sizes[bucket_index]
    buckets[bucket_index, current_size] = vertex_index
    bucket_sizes[bucket_index] += 1
