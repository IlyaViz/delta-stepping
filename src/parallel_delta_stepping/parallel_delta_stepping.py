import math
from atexit import register
from traceback import print_exc
from numpy import ndarray, dtype
from multiprocessing import cpu_count, shared_memory, Lock, Pool
from multiprocessing.synchronize import Lock as LockType
from src.const.dtype import INT_DTYPE, FLOAT_DTYPE, FLOAT_TYPE
from src.shared_delta_stepping.shared_delta_stepping_functions import (
    add_to_bucket,
    prepare_ndarrays,
    validate_and_prepare_variables,
)


MIN_VERTICES_PER_PROCESS = 25

distances_lock_global = None
buckets_lock_global = None

existing_shm_neighbours = None
existing_shm_distances = None
existing_shm_previous = None
existing_shm_weights = None
existing_shm_buckets = None
existing_shm_in_bucket = None
existing_shm_bucket_sizes = None

neighbours_global = None
distances_global = None
previous_global = None
weights_global = None
buckets_global = None
in_bucket_global = None
bucket_sizes_global = None


def parallel_delta_stepping(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex_index: int,
    delta: float = -1,
    processes_count: int = cpu_count(),
) -> list[float]:
    if processes_count <= 0:
        raise ValueError("Processes count must be a positive integer.")

    vertices_length, max_degree, max_buckets, delta = validate_and_prepare_variables(
        neighbours, weights, source_vertex_index, delta
    )

    try:
        shm_list = []

        shm_neighbours = shared_memory.SharedMemory(
            create=True, size=vertices_length * max_degree * INT_DTYPE.itemsize
        )
        shm_list.append(shm_neighbours)

        shm_distances = shared_memory.SharedMemory(
            create=True, size=vertices_length * FLOAT_DTYPE.itemsize
        )
        shm_list.append(shm_distances)

        shm_previous = shared_memory.SharedMemory(
            create=True, size=vertices_length * INT_DTYPE.itemsize
        )
        shm_list.append(shm_previous)

        shm_weights = shared_memory.SharedMemory(
            create=True, size=vertices_length * max_degree * FLOAT_DTYPE.itemsize
        )
        shm_list.append(shm_weights)

        shm_buckets = shared_memory.SharedMemory(
            create=True,
            size=vertices_length * max_buckets * INT_DTYPE.itemsize,
        )
        shm_list.append(shm_buckets)

        shm_in_bucket = shared_memory.SharedMemory(
            create=True, size=max_buckets * vertices_length * dtype(bool).itemsize
        )
        shm_list.append(shm_in_bucket)

        shm_buckets_sizes = shared_memory.SharedMemory(
            create=True, size=max_buckets * INT_DTYPE.itemsize
        )
        shm_list.append(shm_buckets_sizes)

        buffers = [
            shm_neighbours.buf,
            shm_distances.buf,
            shm_previous.buf,
            shm_weights.buf,
            shm_buckets.buf,
            shm_in_bucket.buf,
            shm_buckets_sizes.buf,
        ]
        (
            _,
            shared_distances,
            shared_previous,
            _,
            shared_buckets,
            shared_in_bucket,
            shared_bucket_sizes,
        ) = prepare_ndarrays(
            vertices_length,
            max_degree,
            max_buckets,
            buffers=buffers,
            set_defaults=True,
            source_vertex_index=source_vertex_index,
            neighbours=neighbours,
            weights=weights,
        )

        add_to_bucket(
            delta,
            source_vertex_index,
            shared_distances,
            shared_buckets,
            shared_bucket_sizes,
            max_buckets,
        )

        distances_lock = Lock()
        buckets_lock = Lock()

        with Pool(
            processes_count,
            initializer=init_process,
            initargs=(
                distances_lock,
                buckets_lock,
                shm_neighbours.name,
                shm_distances.name,
                shm_previous.name,
                shm_weights.name,
                shm_buckets.name,
                shm_in_bucket.name,
                shm_buckets_sizes.name,
                max_degree,
                max_buckets,
                vertices_length,
            ),
        ) as pool:
            next_non_empty_bucket_absolute_index = 0
            next_non_empty_bucket_actual_index = (
                next_non_empty_bucket_absolute_index % max_buckets
            )

            while next_non_empty_bucket_absolute_index != -1:
                vertices_in_bucket = shared_bucket_sizes[
                    next_non_empty_bucket_actual_index
                ]
                vertices_per_process = max(
                    MIN_VERTICES_PER_PROCESS, vertices_in_bucket // processes_count
                )
                possible_processes_count = math.ceil(
                    vertices_in_bucket / vertices_per_process
                )

                pool.starmap(
                    process_bucket,
                    [
                        (
                            next_non_empty_bucket_actual_index,
                            i * vertices_per_process,
                            min((i + 1) * vertices_per_process, vertices_in_bucket),
                            max_degree,
                            max_buckets,
                            delta,
                        )
                        for i in range(possible_processes_count)
                    ],
                )

                for i in range(vertices_in_bucket):
                    vertex_index = shared_buckets[next_non_empty_bucket_actual_index, i]
                    shared_in_bucket[
                        next_non_empty_bucket_actual_index, vertex_index
                    ] = False

                shared_bucket_sizes[next_non_empty_bucket_actual_index] = 0

                current_absolute_index = next_non_empty_bucket_absolute_index
                next_non_empty_bucket_absolute_index = -1

                for i in range(max_buckets):
                    absolute_index = current_absolute_index + i
                    actual_index = absolute_index % max_buckets

                    if shared_bucket_sizes[actual_index] > 0:
                        next_non_empty_bucket_absolute_index = absolute_index
                        next_non_empty_bucket_actual_index = actual_index
                        break

        return shared_distances.tolist(), shared_previous.tolist()

    except Exception as e:
        print("An error occurred during parallel delta stepping:")
        print_exc()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()


def init_process(
    distances_lock: LockType,
    buckets_lock: LockType,
    shm_neighbours: str,
    shm_distances: str,
    shm_previous: str,
    shm_weights: str,
    shm_buckets: str,
    shm_in_bucket: str,
    shm_bucket_sizes: str,
    max_degree: int,
    max_buckets: int,
    vertices_length: int,
):
    global distances_lock_global
    global buckets_lock_global

    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_previous
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_bucket_sizes
    global existing_shm_in_bucket

    global neighbours_global
    global previous_global
    global distances_global
    global weights_global
    global buckets_global
    global in_bucket_global
    global bucket_sizes_global

    distances_lock_global = distances_lock
    buckets_lock_global = buckets_lock

    existing_shm_neighbours = shared_memory.SharedMemory(name=shm_neighbours)
    existing_shm_distances = shared_memory.SharedMemory(name=shm_distances)
    existing_shm_previous = shared_memory.SharedMemory(name=shm_previous)
    existing_shm_weights = shared_memory.SharedMemory(name=shm_weights)
    existing_shm_buckets = shared_memory.SharedMemory(name=shm_buckets)
    existing_shm_in_bucket = shared_memory.SharedMemory(name=shm_in_bucket)
    existing_shm_bucket_sizes = shared_memory.SharedMemory(name=shm_bucket_sizes)

    buffers = [
        existing_shm_neighbours.buf,
        existing_shm_distances.buf,
        existing_shm_previous.buf,
        existing_shm_weights.buf,
        existing_shm_buckets.buf,
        existing_shm_in_bucket.buf,
        existing_shm_bucket_sizes.buf,
    ]
    (
        neighbours_global,
        distances_global,
        previous_global,
        weights_global,
        buckets_global,
        in_bucket_global,
        bucket_sizes_global,
    ) = prepare_ndarrays(
        vertices_length,
        max_degree,
        max_buckets,
        buffers=buffers,
    )

    register(close_worker_shm)


def close_worker_shm() -> None:
    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_previous
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_in_bucket
    global existing_shm_bucket_sizes

    for shm in [
        existing_shm_neighbours,
        existing_shm_distances,
        existing_shm_previous,
        existing_shm_weights,
        existing_shm_buckets,
        existing_shm_in_bucket,
        existing_shm_bucket_sizes,
    ]:
        if shm is not None:
            shm.close()


def process_bucket(
    actual_bucket_index: int,
    start_vertex_index: int,
    end_vertex_index: int,
    max_degree: int,
    max_buckets: int,
    delta: float,
) -> None:
    local_buckets = {}
    local_distance_updates = {}
    local_previous_updates = {}
    local_heavy_edges = set()
    vertices_to_process = buckets_global[
        actual_bucket_index, start_vertex_index:end_vertex_index
    ]
    local_buckets[actual_bucket_index] = set(vertices_to_process)

    while local_vertices_to_process := local_buckets[actual_bucket_index]:
        while local_vertices_to_process:
            vertex_index = local_vertices_to_process.pop()

            current_dist = local_distance_updates.get(
                vertex_index, distances_global[vertex_index]
            )

            if int(current_dist // delta) % max_buckets != actual_bucket_index:
                continue

            vertex_neighbour_indexes = neighbours_global[vertex_index]

            for i in range(max_degree):
                neighbour_index = vertex_neighbour_indexes[i]

                if neighbour_index == -1:
                    break

                edge_weight = weights_global[vertex_index, i]
                is_light_edge = edge_weight <= delta

                if not is_light_edge:
                    local_heavy_edges.add((vertex_index, neighbour_index, edge_weight))
                    continue

                if relax_local_neighbour(
                    vertex_index,
                    neighbour_index,
                    edge_weight,
                    distances_global,
                    local_distance_updates,
                    local_previous_updates,
                ):
                    add_to_local_bucket(
                        delta,
                        neighbour_index,
                        local_distance_updates,
                        local_buckets,
                        max_buckets,
                    )

    for vertex_index, neighbour_index, edge_weight in local_heavy_edges:
        if relax_local_neighbour(
            vertex_index,
            neighbour_index,
            edge_weight,
            distances_global,
            local_distance_updates,
            local_previous_updates,
        ):
            add_to_local_bucket(
                delta,
                neighbour_index,
                local_distance_updates,
                local_buckets,
                max_buckets,
            )

    with distances_lock_global:
        for vertex_index, new_distance in local_distance_updates.items():
            if new_distance < distances_global[vertex_index]:
                distances_global[vertex_index] = new_distance
                previous_global[vertex_index] = local_previous_updates[vertex_index]

    with buckets_lock_global:
        for bucket_index, vertices in local_buckets.items():
            for vertex_index in vertices:
                if (
                    int(distances_global[vertex_index] // delta) % max_buckets
                    == bucket_index
                ):
                    if not in_bucket_global[bucket_index, vertex_index]:
                        in_bucket_global[bucket_index, vertex_index] = True

                        current_size = bucket_sizes_global[bucket_index]
                        buckets_global[bucket_index, current_size] = vertex_index
                        bucket_sizes_global[bucket_index] += 1


def add_to_local_bucket(
    delta: float,
    vertex_index: int,
    local_distance_updates: dict[int, float],
    local_buckets: dict[int, set[int]],
    max_local_buckets: int,
) -> None:
    bucket_index = (
        int(local_distance_updates[vertex_index] // delta) % max_local_buckets
    )
    local_bucket = local_buckets.setdefault(bucket_index, set())
    local_bucket.add(vertex_index)


def relax_local_neighbour(
    vertex_index: int,
    neighbour_index: int,
    edge_weight: float,
    distances: ndarray[FLOAT_TYPE],
    local_distance_updates: dict[int, float],
    local_previous_updates: dict[int, int],
) -> bool:
    vertex_distance = local_distance_updates.get(vertex_index, distances[vertex_index])
    neighbour_distance = local_distance_updates.get(
        neighbour_index, distances[neighbour_index]
    )
    new_distance = vertex_distance + edge_weight

    if new_distance < neighbour_distance:
        local_distance_updates[neighbour_index] = new_distance
        local_previous_updates[neighbour_index] = vertex_index

        return True

    return False
