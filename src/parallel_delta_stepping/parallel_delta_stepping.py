import math
from atexit import register
from traceback import print_exc
from numpy import ndarray, int32, float32, dtype
from multiprocessing import cpu_count, shared_memory, Lock, Pool
from src.utils.delta_stepping_params_validator import validate_delta_stepping_params


MIN_VERTICES_PER_PROCESS = 25
INT_DTYPE = dtype(int32)
FLOAT_DTYPE = dtype(float32)


distances_lock_global = None
buckets_lock_global = None

existing_shm_neighbours = None
existing_shm_distances = None
existing_shm_weights = None
existing_shm_buckets = None
existing_shm_in_bucket = None
existing_shm_bucket_sizes = None

neighbours_global = None
distances_global = None
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
) -> None:
    if processes_count <= 0:
        raise ValueError("Processes count must be a positive integer.")

    validate_delta_stepping_params(neighbours, weights, source_vertex_index, delta)

    vertices_length = len(neighbours)
    max_degree = max(len(row) for row in neighbours)
    flattened_weights = [weight for sublist in weights for weight in sublist]

    if delta == -1:
        avg_weight = sum(flattened_weights) / len(flattened_weights)
        delta = avg_weight / 50

    max_weight = max(flattened_weights)
    max_buckets = int(max_weight // delta) + 2

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

        shared_neighbours = ndarray(
            (vertices_length, max_degree), dtype=INT_DTYPE, buffer=shm_neighbours.buf
        )
        shared_distances = ndarray(
            (vertices_length,), dtype=FLOAT_DTYPE, buffer=shm_distances.buf
        )
        shared_weights = ndarray(
            (vertices_length, max_degree), dtype=FLOAT_DTYPE, buffer=shm_weights.buf
        )
        shared_buckets = ndarray(
            (max_buckets, vertices_length),
            dtype=INT_DTYPE,
            buffer=shm_buckets.buf,
        )
        shared_in_bucket = ndarray(
            (max_buckets, vertices_length), dtype=bool, buffer=shm_in_bucket.buf
        )
        shared_bucket_sizes = ndarray(
            (max_buckets,), dtype=INT_DTYPE, buffer=shm_buckets_sizes.buf
        )

        shared_neighbours.fill(-1)
        shared_distances.fill(float("inf"))
        shared_bucket_sizes.fill(0)

        shared_distances[source_vertex_index] = 0.0

        padded_neighbours = [row + [-1] * (max_degree - len(row)) for row in neighbours]
        padded_weights = [row + [0.0] * (max_degree - len(row)) for row in weights]

        shared_neighbours[:, :] = padded_neighbours
        shared_weights[:, :] = padded_weights

        add_to_bucket(
            delta,
            source_vertex_index,
            shared_distances,
            shared_buckets,
            shared_bucket_sizes,
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

        return shared_distances.tolist()

    except Exception as e:
        print("An error occurred during parallel delta stepping:")
        print_exc()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()


def init_process(
    distances_lock,
    buckets_lock,
    shm_neighbours,
    shm_distances,
    shm_weights,
    shm_buckets,
    shm_in_bucket,
    shm_bucket_sizes,
    max_degree,
    max_buckets,
    vertices_length,
):
    global distances_lock_global
    global buckets_lock_global

    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_bucket_sizes
    global existing_shm_in_bucket
    global neighbours_global
    global distances_global
    global weights_global
    global buckets_global
    global in_bucket_global
    global bucket_sizes_global

    distances_lock_global = distances_lock
    buckets_lock_global = buckets_lock

    existing_shm_neighbours = shared_memory.SharedMemory(name=shm_neighbours)
    existing_shm_distances = shared_memory.SharedMemory(name=shm_distances)
    existing_shm_weights = shared_memory.SharedMemory(name=shm_weights)
    existing_shm_buckets = shared_memory.SharedMemory(name=shm_buckets)
    existing_shm_in_bucket = shared_memory.SharedMemory(name=shm_in_bucket)
    existing_shm_bucket_sizes = shared_memory.SharedMemory(name=shm_bucket_sizes)

    neighbours_global = ndarray(
        (vertices_length, max_degree),
        dtype=INT_DTYPE,
        buffer=existing_shm_neighbours.buf,
    )
    distances_global = ndarray(
        (vertices_length,), dtype=FLOAT_DTYPE, buffer=existing_shm_distances.buf
    )
    weights_global = ndarray(
        (vertices_length, max_degree),
        dtype=FLOAT_DTYPE,
        buffer=existing_shm_weights.buf,
    )
    buckets_global = ndarray(
        (max_buckets, vertices_length),
        dtype=INT_DTYPE,
        buffer=existing_shm_buckets.buf,
    )
    in_bucket_global = ndarray(
        (max_buckets, vertices_length), dtype=bool, buffer=existing_shm_in_bucket.buf
    )
    bucket_sizes_global = ndarray(
        (max_buckets,),
        dtype=INT_DTYPE,
        buffer=existing_shm_bucket_sizes.buf,
    )

    register(close_worker_shm)


def close_worker_shm():
    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_in_bucket
    global existing_shm_bucket_sizes

    for shm in [
        existing_shm_neighbours,
        existing_shm_distances,
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

            if (
                current_dist == float("inf")
                or int(current_dist // delta) % max_buckets != actual_bucket_index
            ):
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

                if relax_neighbour(
                    vertex_index,
                    neighbour_index,
                    edge_weight,
                    distances_global,
                    local_distance_updates,
                ):
                    add_to_local_bucket(
                        delta,
                        neighbour_index,
                        local_distance_updates,
                        local_buckets,
                        max_buckets,
                    )

    for vertex_index, neighbour_index, edge_weight in local_heavy_edges:
        if relax_neighbour(
            vertex_index,
            neighbour_index,
            edge_weight,
            distances_global,
            local_distance_updates,
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

    with buckets_lock_global:
        for bucket_index, vertices in local_buckets.items():
            for vertex_index in vertices:
                if (
                    distances_global[vertex_index] == float("inf")
                    or int(distances_global[vertex_index] // delta) % max_buckets
                    == bucket_index
                ):
                    if not in_bucket_global[bucket_index, vertex_index]:
                        in_bucket_global[bucket_index, vertex_index] = True

                        current_size = bucket_sizes_global[bucket_index]
                        buckets_global[bucket_index, current_size] = vertex_index
                        bucket_sizes_global[bucket_index] += 1


def relax_neighbour(
    vertex_index: int,
    neighbour_index: int,
    edge_weight: float,
    distances: ndarray[FLOAT_DTYPE],
    local_distance_updates: dict[int, float],
) -> bool:
    vertex_distance = local_distance_updates.get(vertex_index, distances[vertex_index])
    neighbour_distance = local_distance_updates.get(
        neighbour_index, distances[neighbour_index]
    )
    new_distance = vertex_distance + edge_weight

    if new_distance < neighbour_distance:
        local_distance_updates[neighbour_index] = new_distance
        return True

    return False


def add_to_bucket(
    delta,
    vertex: int,
    distances: ndarray[FLOAT_DTYPE],
    buckets: ndarray[INT_DTYPE],
    bucket_sizes: ndarray[INT_DTYPE],
) -> None:
    max_buckets = len(bucket_sizes)
    bucket_index = int(distances[vertex] // delta) % max_buckets

    current_size = bucket_sizes[bucket_index]
    buckets[bucket_index, current_size] = vertex

    bucket_sizes[bucket_index] += 1


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
