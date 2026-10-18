from abc import ABC, abstractmethod

from ..environments.mdpenvironment import MDPEnvironment


class MDPSolver[State, Action](ABC):
    def __init__(self, env: MDPEnvironment[State, Action]) -> None:
        self.env = env

    @abstractmethod
    def solve(self) -> None:
        """Compute the optimal policy and value function."""
        ...

    @abstractmethod
    def get_action(self, state: State, time_step: int | None = None) -> Action:
        """Return the selected action for a given state."""
        ...

    @abstractmethod
    def get_value(self, state: State, time_step: int | None = None) -> float:
        """Return the value of a given state."""
        ...
