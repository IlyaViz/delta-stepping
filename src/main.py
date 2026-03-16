from os import cpu_count

from src.parallel_delta_stepping.parallel_delta_stepping import parallel_delta_stepping
from src.generator.weighted_graph_generator import generate_weighted_graph
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
import time


if __name__ == "__main__":
    graph = generate_weighted_graph(200, 1000)

    start_parallel = time.time()
    parallel_delta_stepping(graph, graph.vertices[0], 1, 16)
    end_parallel = time.time()

    start_sequential = time.time()
    sequential_delta_stepping(graph, graph.vertices[0], 1)
    end_sequential = time.time()

    print(f"Parallel time: {end_parallel - start_parallel:.4f} seconds")
    print(f"Sequential time: {end_sequential - start_sequential:.4f} seconds")
