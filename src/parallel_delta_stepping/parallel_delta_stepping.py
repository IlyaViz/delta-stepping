import traceback
from numpy import int64, ndarray, float64, dtype
from multiprocessing import cpu_count, shared_memory, Manager, Lock, Pool
from multiprocessing.synchronize import Lock as LockType
from src.weighted_graph.weighted_graph import WeightedGraph
from src.weighted_graph.weighted_vertex import WeightedVertex


def parallel_delta_stepping(
    graph: WeightedGraph,
    source_vertex: WeightedVertex,
    delta: float,
    processes_count: int = cpu_count(),
) -> None:
    try:
        vertices_length = len(graph.vertices)
        max_degree = max(
            len(graph.get_vertex_edges(vertex)) for vertex in graph.vertices
        )
        max_weight = max(
            edge.weight
            for vertex in graph.vertices
            for edge in graph.get_vertex_edges(vertex)
        )
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
            create=True, size=vertices_length * max_buckets * dtype(int64).itemsize
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
            (max_buckets, vertices_length), dtype=int64, buffer=shm_buckets.buf
        )
        shared_bucket_sizes = ndarray(
            (max_buckets,), dtype=int64, buffer=shm_buckets_sizes.buf
        )

        shared_neighbours.fill(-1)
        shared_distances.fill(float("inf"))
        shared_bucket_sizes.fill(0)

        source_vertex_index = graph.vertices.index(source_vertex)
        shared_distances[source_vertex_index] = 0.0

        for n, vertex in enumerate(graph.vertices):
            vertex_edges = graph.get_vertex_edges(vertex)

            for e, edge in enumerate(vertex_edges):
                neighbour_vertex = graph.get_vertex_neighbour_by_edge(vertex, edge)
                neighbour_vertex_index = graph.vertices.index(neighbour_vertex)

                shared_neighbours[n, e] = neighbour_vertex_index
                shared_weights[n, e] = edge.weight

        add_to_bucket(
            delta,
            source_vertex_index,
            shared_distances,
            shared_buckets,
            shared_bucket_sizes,
        )

        with Manager() as manager:
            distances_lock = manager.Lock()
            buckets_lock = manager.Lock()

            args_template = (
                processes_count,
                vertices_length,
                max_degree,
                max_buckets,
                delta,
                shm_neighbours.name,
                shm_distances.name,
                shm_weights.name,
                shm_buckets.name,
                shm_buckets_sizes.name,
                distances_lock,
                buckets_lock,
            )

            with Pool(processes_count) as pool:
                current_bucket_index = 0

                while any(bucket_size != 0 for bucket_size in shared_bucket_sizes):
                    actual_bucket_index = current_bucket_index % max_buckets

                    if shared_bucket_sizes[actual_bucket_index] > 0:
                        tasks = [
                            (current_bucket_index,) + args_template
                            for _ in range(processes_count)
                        ]
                        pool.starmap(process_bucket, tasks)

                    current_bucket_index += 1

            for n, vertex in enumerate(graph.vertices):
                vertex.distance = shared_distances[n]
    except Exception as e:
        print("An error occurred during parallel delta stepping:")
        traceback.print_exc()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()


def relax_neighbour(
    vertex_index: int,
    neighbour_index: int,
    edge_weight: float,
    distances: ndarray[float64],
) -> bool:
    new_distance = distances[vertex_index] + edge_weight

    if new_distance < distances[neighbour_index]:
        distances[neighbour_index] = new_distance

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
    distances: ndarray[float64],
    local_buckets: list[set[int]],
    max_local_buckets: int,
) -> None:
    bucket_index = int(distances[vertex_index] // delta) % max_local_buckets
    buckets_length = len(local_buckets)

    for _ in range(bucket_index - buckets_length + 1):
        local_buckets.append(set())

    local_buckets[bucket_index].add(vertex_index)


def process_bucket(
    absolute_bucket_index: int,
    total_proccesses: int,
    total_vertices: int,
    max_degree: int,
    max_buckets: int,
    delta: float,
    shm_neighbours_name: str,
    shm_distances_name: str,
    shm_weights_name: str,
    shm_buckets_name: str,
    shm_bucket_sizes_name: str,
    distances_lock: LockType,
    buckets_lock: LockType,
) -> None:
    existing_shm_neighbours = shared_memory.SharedMemory(name=shm_neighbours_name)
    existing_shm_distances = shared_memory.SharedMemory(name=shm_distances_name)
    existing_shm_weights = shared_memory.SharedMemory(name=shm_weights_name)
    existing_shm_buckets = shared_memory.SharedMemory(name=shm_buckets_name)
    existing_shm_bucket_sizes = shared_memory.SharedMemory(name=shm_bucket_sizes_name)

    neighbours = ndarray(
        (total_vertices, max_degree), dtype=int64, buffer=existing_shm_neighbours.buf
    )
    distances = ndarray(
        (total_vertices,), dtype=float64, buffer=existing_shm_distances.buf
    )
    weights = ndarray(
        (total_vertices, max_degree), dtype=float64, buffer=existing_shm_weights.buf
    )
    buckets = ndarray(
        (max_buckets, total_vertices),
        dtype=int64,
        buffer=existing_shm_buckets.buf,
    )
    bucket_sizes = ndarray(
        (max_buckets,), dtype=int64, buffer=existing_shm_bucket_sizes.buf
    )

    actual_bucket_index = absolute_bucket_index % max_buckets

    vertices_per_process = max(1, total_vertices // total_proccesses)
    local_buckets = []
    local_distances = distances.copy()
    local_heavy_edges = set()
    local_vertices_to_process_set = set()
    processing = True

    while processing:
        local_vertices_to_process_set_length = len(local_vertices_to_process_set)

        if local_vertices_to_process_set_length < vertices_per_process:
            additional_vertices_to_process_count = (
                vertices_per_process - local_vertices_to_process_set_length
            )

            extra_vertices_to_process = []

            with buckets_lock:
                available_extra_vertices = bucket_sizes[actual_bucket_index]
                extra_vertices_to_process_count = min(
                    additional_vertices_to_process_count, available_extra_vertices
                )

                if extra_vertices_to_process_count > 0:
                    start_index = (
                        available_extra_vertices - extra_vertices_to_process_count
                    )

                    extra_vertices_to_process = buckets[
                        actual_bucket_index,
                        start_index:available_extra_vertices,
                    ]

                    local_vertices_to_process_set.update(extra_vertices_to_process)

                    bucket_sizes[
                        actual_bucket_index
                    ] -= extra_vertices_to_process_count

        if not local_vertices_to_process_set:
            processing = False
            continue

        for vertex_index in local_vertices_to_process_set:
            if int(local_distances[vertex_index] // delta) != absolute_bucket_index:
                continue

            vertex_neighbour_indexes = neighbours[vertex_index]

            for i in range(max_degree):
                neighbour_index = vertex_neighbour_indexes[i]

                if neighbour_index == -1:
                    break

                edge_weight = weights[vertex_index, i]
                is_light_edge = edge_weight <= delta

                if not is_light_edge:
                    local_heavy_edges.add((vertex_index, neighbour_index, edge_weight))
                    continue

                if relax_neighbour(
                    vertex_index, neighbour_index, edge_weight, local_distances
                ):
                    add_to_local_bucket(
                        delta,
                        neighbour_index,
                        local_distances,
                        local_buckets,
                        max_buckets,
                    )

        local_vertices_to_process_set.clear()

        if actual_bucket_index < len(local_buckets):
            for vertex_index in local_buckets[actual_bucket_index]:
                local_vertices_to_process_set.add(vertex_index)

            local_buckets[actual_bucket_index].clear()

    for vertex_index, neighbour_index, edge_weight in local_heavy_edges:
        if relax_neighbour(vertex_index, neighbour_index, edge_weight, local_distances):
            add_to_local_bucket(
                delta, neighbour_index, local_distances, local_buckets, max_buckets
            )

    with distances_lock:
        for vertex_index in range(total_vertices):
            if local_distances[vertex_index] < distances[vertex_index]:
                distances[vertex_index] = local_distances[vertex_index]

    with buckets_lock:
        for bucket_index, bucket in enumerate(local_buckets):
            for vertex_index in bucket:
                current_size = bucket_sizes[bucket_index]
                buckets[bucket_index, current_size] = vertex_index

                bucket_sizes[bucket_index] += 1
