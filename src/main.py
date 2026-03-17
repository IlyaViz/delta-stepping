import time
from os import cpu_count
from src.parallel_delta_stepping.parallel_delta_stepping import parallel_delta_stepping
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)


if __name__ == "__main__":
    vertex_count = 20000
    edge_count = 20000000
    delta = 1
    min_weight = 1
    max_weight = 100

    create_graph_start_time = time.time()
    neighbours, weights = generate_weighted_graph_with_default_types(
        vertex_count, edge_count, min_weight, max_weight
    )
    create_graph_end_time = time.time()

    print(
        f"Graph creation time: {create_graph_end_time - create_graph_start_time:.4f} seconds"
    )

    print("\nStarted")

    start_sequential = time.time()
    sequential_delta_stepping(neighbours, weights, 0, delta)
    end_sequential = time.time()

    start_parallel_1 = time.time()
    parallel_delta_stepping(neighbours, weights, 0, delta, 1)
    end_parallel_1 = time.time()

    start_parallel_8 = time.time()
    parallel_delta_stepping(neighbours, weights, 0, delta, 8)
    end_parallel_8 = time.time()

    print("\nFinished")
    print(f"Sequential time: {end_sequential - start_sequential:.4f} seconds")
    print(f"Parallel time (1 thread): {end_parallel_1 - start_parallel_1:.4f} seconds")
    print(f"Parallel time (8 threads): {end_parallel_8 - start_parallel_8:.4f} seconds")
