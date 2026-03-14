from dataclasses import dataclass
from src.weighted_graph.weighted_vertex import WeightedVertex

@dataclass(frozen=True)
class WeightedEdge:
    source: WeightedVertex
    target: WeightedVertex
    weight: int
