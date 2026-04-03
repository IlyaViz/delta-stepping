import time
import matplotlib.pyplot as plt
import math
from colorama import Fore, Style
from src.parallel_delta_stepping.parallel_delta_stepping import parallel_delta_stepping
from src.sequential_delta_stepping.sequential_delta_stepping import (
    sequential_delta_stepping,
)
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from src.const.type_error import ABS_TOL, REL_TOL


def perform_delta_stepping_analysis(
    vertex_options: list[int],
    edge_ratio_options: list[float],
    deltas: list[float],
    cpu_count: list[int],
    output_folder: str,
    min_weight: int = 1,
    max_weight: int = 100,
    retries: int = 3,
) -> None:
    analysis_start = time.time()

    with open(f"{output_folder}/summary.txt", "w") as f:
        f.write("")

    step_count = (
        len(vertex_options) * len(edge_ratio_options) * len(deltas) * len(cpu_count)
    )
    step = 0

    for delta in deltas:
        fig, axes = plt.subplots(len(vertex_options), 2, figsize=(15, 10))

        for vertex_index, vertex_count in enumerate(vertex_options):
            for cpus in cpu_count:
                cpus_speedup_results = []
                cpus_efficiency_results = []

                for edge_ratio in edge_ratio_options:
                    step += 1
                    print(
                        f"{Fore.GREEN}Step {step}/{step_count} started{Style.RESET_ALL}"
                    )

                    neighbours, weights = generate_weighted_graph_with_default_types(
                        vertex_count,
                        int(vertex_count * edge_ratio),
                        min_weight,
                        max_weight,
                    )

                    total_sequential_time = 0
                    total_parallel_time = 0

                    for _ in range(retries):
                        start = time.time()
                        sequential_distances = sequential_delta_stepping(
                            neighbours, weights, 0, delta
                        )
                        end = time.time()
                        total_sequential_time += end - start

                        start = time.time()
                        parallel_distanaces = parallel_delta_stepping(
                            neighbours, weights, 0, delta, cpus
                        )
                        end = time.time()
                        total_parallel_time += end - start

                        for vertex in range(vertex_count):
                            if not math.isclose(
                                sequential_distances[vertex],
                                parallel_distanaces[vertex],
                                abs_tol=ABS_TOL,
                                rel_tol=REL_TOL,
                            ):
                                with open(f"{output_folder}/summary.txt", "w") as f:
                                    f.write(
                                        f"Distances {sequential_distances[vertex]} and {parallel_distanaces[vertex]} do not match for vertex {vertex} in graph with {vertex_count} vertices and edge ratio {edge_ratio}\n"
                                    )
                                raise ValueError(
                                    f"Distances {sequential_distances[vertex]} and {parallel_distanaces[vertex]} do not match for vertex {vertex} in graph with {vertex_count} vertices and edge ratio {edge_ratio}"
                                )

                    average_sequential_time = total_sequential_time / retries
                    average_parallel_time = total_parallel_time / retries

                    with open(f"{output_folder}/summary.txt", "a") as f:
                        f.write(
                            f"Vertex Count: {vertex_count}, Edge Ratio: {edge_ratio}, Delta: {delta}, CPUs: {cpus}\n"
                        )
                        f.write(
                            f"Average Sequential Time: {average_sequential_time:.4f} seconds\n"
                        )
                        f.write(
                            f"Average Parallel Time: {average_parallel_time:.4f} seconds\n"
                        )
                        f.write("\n")

                    speedup = average_sequential_time / average_parallel_time
                    efficiency = speedup / cpus

                    cpus_speedup_results.append(speedup)
                    cpus_efficiency_results.append(efficiency)

                axes[vertex_index][0].plot(
                    edge_ratio_options,
                    cpus_speedup_results,
                    marker="o",
                    label=f"{cpus} Threads",
                )
                axes[vertex_index][0].set_title(f"Speedup for {vertex_count} vertices")
                axes[vertex_index][0].set_xlabel("Edge Ratio")
                axes[vertex_index][0].set_ylabel("Speedup")
                axes[vertex_index][0].legend()

                axes[vertex_index][1].plot(
                    edge_ratio_options,
                    cpus_efficiency_results,
                    marker="o",
                    label=f"{cpus} CPUs",
                )
                axes[vertex_index][1].set_title(
                    f"Efficiency for {vertex_count} vertices"
                )
                axes[vertex_index][1].set_xlabel("Edge Ratio")
                axes[vertex_index][1].set_ylabel("Efficiency")
                axes[vertex_index][1].legend()

        fig.suptitle(f"Delta: {delta}")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/delta_{delta}_analysis.png")

    analysis_end = time.time()

    with open(f"{output_folder}/summary.txt", "a") as f:
        f.write(f"Delta Stepping Analysis Summary\n")
        f.write(f"Total analysis time: {analysis_end - analysis_start:.2f} seconds\n")
