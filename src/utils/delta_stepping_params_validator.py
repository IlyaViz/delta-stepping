def validate_delta_stepping_params(
    neighbours: list[list[int]],
    weights: list[list[float]],
    source_vertex: int,
    delta: float,
) -> None:
    if len(neighbours) <= 1:
        raise ValueError("Graph must have at least 2 vertices.")

    if len(neighbours) != len(weights):
        raise ValueError("Neighbours and weights must have the same length.")

    for vertex_weights in weights:
        for weight in vertex_weights:
            if weight < 0:
                raise ValueError(
                    "Graph contains negative weights, which is not allowed."
                )

    if source_vertex < 0 or source_vertex >= len(neighbours):
        raise ValueError("Source vertex index is out of bounds.")

    if delta <= 0 and delta != -1:
        raise ValueError(
            "Delta must be a positive number or -1 for automatic calculation."
        )
