import time
from src.delta_stepping_analyzer.delta_stepping_analyzer import (
    perform_delta_stepping_analysis,
)
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.parallel_delta_stepping.parallel_delta_stepping import (
    parallel_delta_stepping,
)


if __name__ == "__main__":
    # vertex_options = [1000, 5000, 10000]
    # edge_ratio_options = [10, 500, 1000]
    # deltas = [1, 10, 20, 50]
    # cpu_options = [2, 4, 8, 16]
    # min_weight = 1
    # max_weight = 100

    vertex_options = [2, 1000]
    edge_ratio_options = [500, 1000]
    deltas = [1, 2]
    cpu_options = [4, 8, 16]
    min_weight = 1
    max_weight = 100

    perform_delta_stepping_analysis(
        vertex_options=vertex_options,
        edge_ratio_options=edge_ratio_options,
        deltas=deltas,
        cpu_count=cpu_options,
        output_folder="docs/analysis_results",
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # neighbours, weights = generate_weighted_graph_with_default_types(1, 500, 1, 100)

    # start = time.time()
    # sequential_delta_stepping(neighbours, weights, 0, 1)
    # end = time.time()

    # print(f"Sequential delta stepping time: {end - start:.4f} seconds")

    # start = time.time()
    # parallel_delta_stepping(neighbours, weights, 0, 10, 4)
    # end = time.time()
    # print(f"Parallel delta stepping time: {end - start:.4f} seconds")
