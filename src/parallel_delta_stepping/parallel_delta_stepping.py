from atexit import register
from traceback import print_exc
from numpy import int64, ndarray, float64, dtype
from multiprocessing import cpu_count, shared_memory, Lock, Pool

distances_lock_global = None
buckets_lock_global = None

existing_shm_neighbours = None
existing_shm_distances = None
existing_shm_weights = None
existing_shm_buckets = None
existing_shm_bucket_sizes = None

neighbours_global = None
distances_global = None
weights_global = None
buckets_global = None
bucket_sizes_global = None


def parallel_delta_stepping(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex_index: int,
    delta: float,
    processes_count: int = cpu_count(),
) -> None:
    try:
        vertices_length = len(neighbours)
        max_degree = max(len(row) for row in neighbours)
        max_weight = max(max(row, default=0.0) for row in weights)
        max_buckets = int(max_weight // delta) + 1

        shm_list = []

        shm_neighbours = shared_memory.SharedMemory(
            create=True, size=vertices_length * max_degree * dtype(int64).itemsize
        )
        shm_list.append(shm_neighbours)

        shm_distances = shared_memory.SharedMemory(
            create=True, size=vertices_length * dtype(float64).itemsize
        )
        shm_list.append(shm_distances)

        shm_weights = shared_memory.SharedMemory(
            create=True, size=vertices_length * max_degree * dtype(float64).itemsize
        )
        shm_list.append(shm_weights)

        shm_buckets = shared_memory.SharedMemory(
            create=True,
            size=vertices_length
            * max_buckets
            * processes_count
            * dtype(int64).itemsize,
        )
        shm_list.append(shm_buckets)

        shm_buckets_sizes = shared_memory.SharedMemory(
            create=True, size=max_buckets * dtype(int64).itemsize
        )
        shm_list.append(shm_buckets_sizes)

        shared_neighbours = ndarray(
            (vertices_length, max_degree), dtype=int64, buffer=shm_neighbours.buf
        )
        shared_distances = ndarray(
            (vertices_length,), dtype=float64, buffer=shm_distances.buf
        )
        shared_weights = ndarray(
            (vertices_length, max_degree), dtype=float64, buffer=shm_weights.buf
        )
        shared_buckets = ndarray(
            (max_buckets, vertices_length * processes_count),
            dtype=int64,
            buffer=shm_buckets.buf,
        )
        shared_bucket_sizes = ndarray(
            (max_buckets,), dtype=int64, buffer=shm_buckets_sizes.buf
        )

        shared_neighbours.fill(-1)
        shared_distances.fill(float("inf"))
        shared_bucket_sizes.fill(0)

        shared_distances[source_vertex_index] = 0.0

        for row in neighbours:
            row.extend([-1] * (max_degree - len(row)))

        for row in weights:
            row.extend([0.0] * (max_degree - len(row)))

        shared_neighbours[:, :] = neighbours
        shared_weights[:, :] = weights

        add_to_bucket(
            delta,
            source_vertex_index,
            shared_distances,
            shared_buckets,
            shared_bucket_sizes,
        )

        distances_lock = Lock()
        buckets_lock = Lock()

        total_vertices_process = 0
        steps = 0

        with Pool(
            processes_count,
            initializer=init_pool_locks,
            initargs=(
                distances_lock,
                buckets_lock,
                shm_neighbours.name,
                shm_distances.name,
                shm_weights.name,
                shm_buckets.name,
                shm_buckets_sizes.name,
                max_degree,
                max_buckets,
                processes_count,
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
                vertices_per_process = (
                    vertices_in_bucket + processes_count - 1
                ) // processes_count

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
                        for i in range(processes_count)
                    ],
                )

                total_vertices_process += vertices_in_bucket
                steps += 1

                shared_bucket_sizes[next_non_empty_bucket_actual_index] = 0

                for i in range(max_buckets):
                    absolute_index = next_non_empty_bucket_absolute_index + i
                    actual_index = absolute_index % max_buckets

                    if shared_bucket_sizes[actual_index] > 0:
                        next_non_empty_bucket_absolute_index = absolute_index
                        next_non_empty_bucket_actual_index = actual_index
                        break

                    next_non_empty_bucket_absolute_index = -1

        print(
            f"Average vertices per process: {total_vertices_process / steps / processes_count:.2f}"
        )

        return shared_distances.tolist()

    except Exception as e:
        print("An error occurred during parallel delta stepping:")
        print_exc()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()


def init_pool_locks(
    distances_lock,
    buckets_lock,
    shm_neighbours,
    shm_distances,
    shm_weights,
    shm_buckets,
    shm_bucket_sizes,
    max_degree,
    max_buckets,
    processes_count,
    vertices_length,
):
    global distances_lock_global
    global buckets_lock_global

    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_bucket_sizes

    global neighbours_global
    global distances_global
    global weights_global
    global buckets_global
    global bucket_sizes_global

    distances_lock_global = distances_lock
    buckets_lock_global = buckets_lock

    existing_shm_neighbours = shared_memory.SharedMemory(name=shm_neighbours)
    existing_shm_distances = shared_memory.SharedMemory(name=shm_distances)
    existing_shm_weights = shared_memory.SharedMemory(name=shm_weights)
    existing_shm_buckets = shared_memory.SharedMemory(name=shm_buckets)
    existing_shm_bucket_sizes = shared_memory.SharedMemory(name=shm_bucket_sizes)

    neighbours_global = ndarray(
        (vertices_length, max_degree),
        dtype=int64,
        buffer=existing_shm_neighbours.buf,
    )
    distances_global = ndarray(
        (vertices_length,), dtype=float64, buffer=existing_shm_distances.buf
    )
    weights_global = ndarray(
        (vertices_length, max_degree),
        dtype=float64,
        buffer=existing_shm_weights.buf,
    )
    buckets_global = ndarray(
        (max_buckets, vertices_length * processes_count),
        dtype=int64,
        buffer=existing_shm_buckets.buf,
    )
    bucket_sizes_global = ndarray(
        (max_buckets,),
        dtype=int64,
        buffer=existing_shm_bucket_sizes.buf,
    )

    register(close_worker_shm)


def close_worker_shm():
    global existing_shm_neighbours
    global existing_shm_distances
    global existing_shm_weights
    global existing_shm_buckets
    global existing_shm_bucket_sizes

    for shm in [
        existing_shm_neighbours,
        existing_shm_distances,
        existing_shm_weights,
        existing_shm_buckets,
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
    local_buckets = {i: set() for i in range(max_buckets)}
    local_distance_updates = {}
    local_heavy_edges = set()
    vertices_to_process = buckets_global[
        actual_bucket_index, start_vertex_index:end_vertex_index
    ]
    local_buckets[actual_bucket_index] = set(vertices_to_process)

    while local_vertices_to_process := local_buckets[actual_bucket_index]:
        for vertex_index in set(local_vertices_to_process):
            local_buckets[actual_bucket_index].remove(vertex_index)

            if (
                distances_global[vertex_index] == float("inf")
                or int(distances_global[vertex_index] // delta) % max_buckets
                != actual_bucket_index
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
                    current_size = bucket_sizes_global[bucket_index]
                    buckets_global[bucket_index, current_size] = vertex_index
                    bucket_sizes_global[bucket_index] += 1


def relax_neighbour(
    vertex_index: int,
    neighbour_index: int,
    edge_weight: float,
    distances: ndarray[float64],
    local_distance_updates: dict[int, float],
) -> bool:
    neighbour_distance = local_distance_updates.get(
        neighbour_index, distances[neighbour_index]
    )
    new_distance = distances[vertex_index] + edge_weight

    if new_distance < neighbour_distance:
        local_distance_updates[neighbour_index] = new_distance
        return True

    return False


def add_to_bucket(
    delta,
    vertex: int,
    distances: ndarray[float64],
    buckets: ndarray[int64],
    bucket_sizes: ndarray[int64],
) -> None:
    max_buckets = len(bucket_sizes)
    bucket_index = int(distances[vertex] // delta) % max_buckets

    current_size = bucket_sizes[bucket_index]
    buckets[bucket_index, current_size] = vertex

    bucket_sizes[bucket_index] += 1


def add_to_local_bucket(
    delta,
    vertex_index: int,
    local_distance_updates: dict[int, float],
    local_buckets: dict[int, set[int]],
    max_local_buckets: int,
) -> None:
    bucket_index = (
        int(local_distance_updates[vertex_index] // delta) % max_local_buckets
    )
    local_buckets[bucket_index].add(vertex_index)
