import pytest
from src.utils.delta_stepping_params_validator import (
    validate_delta_stepping_params,
)


def test_validate_incorrect_neighbours_length(test_graph):
    neighbours, weights, _ = test_graph

    neighbours = []

    with pytest.raises(ValueError, match="Graph must have at least 2 vertices."):
        validate_delta_stepping_params(neighbours, weights, 0, 1)


def test_validate_incorrect_neighbours(test_graph):
    neighbours, weights, _ = test_graph

    del neighbours[0]

    with pytest.raises(
        ValueError, match="Neighbours and weights must have the same length."
    ):
        validate_delta_stepping_params(neighbours, weights, 0, 1)


def test_validate_incorrect_weights(test_graph):
    neighbours, weights, _ = test_graph

    del weights[0]

    with pytest.raises(
        ValueError, match="Neighbours and weights must have the same length."
    ):
        validate_delta_stepping_params(neighbours, weights, 0, 1)


def test_validate_incorrect_source_vertex(test_graph):
    neighbours, weights, _ = test_graph

    with pytest.raises(ValueError, match="Source vertex index is out of bounds."):
        validate_delta_stepping_params(neighbours, weights, -1, 1)

    with pytest.raises(ValueError, match="Source vertex index is out of bounds."):
        validate_delta_stepping_params(neighbours, weights, len(neighbours), 1)


@pytest.mark.parametrize("delta", [0, -2])
def test_validate_incorrect_delta(test_graph, delta):
    neighbours, weights, _ = test_graph

    with pytest.raises(
        ValueError,
        match="Delta must be a positive number or -1 for automatic calculation.",
    ):
        validate_delta_stepping_params(neighbours, weights, 0, delta)


def test_validate_incorrect_weights_values(test_graph):
    neighbours, weights, _ = test_graph

    weights[0][0] = -1

    with pytest.raises(
        ValueError,
        match="Graph contains negative weights, which is not allowed.",
    ):
        validate_delta_stepping_params(neighbours, weights, 0, 1)


def test_validate_correct_params(test_graph):
    neighbours, weights, _ = test_graph

    validate_delta_stepping_params(neighbours, weights, 0, 1)
