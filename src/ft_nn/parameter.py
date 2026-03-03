from dataclasses import dataclass


@dataclass(eq=False)
class Parameter:
    data: float = 0.0
    grad: float = 0.0
