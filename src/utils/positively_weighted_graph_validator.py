def validate_positively_weighted_graph(weights: list[list[float]]) -> None:
    for vertex_weights in weights:
        for weight in vertex_weights:
            if weight < 0:
                raise ValueError(
                    "Graph contains negative weights, which is not allowed."
                )
