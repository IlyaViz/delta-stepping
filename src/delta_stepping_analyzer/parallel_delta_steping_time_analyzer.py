import time
import matplotlib.pyplot as plt
import math
from colorama import Fore, Style
from src.generator.weighted_graph_generator import (
    generate_weighted_graph_with_default_types,
)
from src.parallel_delta_stepping.parallel_delta_stepping_time_calculation import (
    parallel_delta_stepping_time_calculation,
)


def perform_parallel_delta_stepping_time_analysis(
    vertex_options: list[int],
    edge_ratio_options: list[float],
    deltas: list[float],
    output_folder: str,
    min_weight: int = 1,
    max_weight: int = 100,
    retries: int = 3,
) -> None:
    analysis_start = time.time()
    fig, axes = plt.subplots(len(vertex_options), 1, figsize=(15, 10))
    step = 0
    step_count = len(vertex_options) * len(edge_ratio_options) * len(deltas)

    for vertex_index, vertex_count in enumerate(vertex_options):
        for delta in deltas:
            sequential_ratios = []

            for edge_ratio in edge_ratio_options:
                step += 1
                print(f"{Fore.GREEN}Step {step}/{step_count} started{Style.RESET_ALL}")

                neighbours, weights = generate_weighted_graph_with_default_types(
                    vertex_count,
                    int(vertex_count * edge_ratio),
                    min_weight,
                    max_weight,
                )

                total_sequential_ratio = 0

                for _ in range(retries):
                    results = parallel_delta_stepping_time_calculation(
                        neighbours, weights, 0
                    )

                    total_sequential_ratio += results[0]

                average_sequential_ratio = total_sequential_ratio / retries
                sequential_ratios.append(average_sequential_ratio)

            axes[vertex_index].plot(
                edge_ratio_options, sequential_ratios, label=f"Delta: {delta}"
            )
            axes[vertex_index].set_title(f"Vertex count: {vertex_count}")
            axes[vertex_index].set_xlabel("Edge ratio")
            axes[vertex_index].set_ylabel("Main sequential ratio")
            axes[vertex_index].legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/parallel_delta_stepping_time_analysis.png")

    analysis_end = time.time()

    with open(f"{output_folder}/summary.txt", "w") as f:
        f.write(f"Analysis completed in {analysis_end - analysis_start} seconds.\n")
